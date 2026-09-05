"""Small MNIST -> USPS DANN demo. Importing this module never trains a model."""
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import MNIST, USPS


@dataclass
class Config:
    seed: int = 7
    epochs: int = 30
    batch_size: int = 128
    source_samples: int | None = 24000  # None uses all available training images.
    target_samples: int | None = None
    learning_rate: float = 1e-3
    adversarial_weight: float = 0.1
    warmup_epochs: int = 3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ImagesOnly(Dataset):
    """The adaptation dataset stores images only: no USPS digit labels."""
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def digit_indices(labels, limit=None, seed=7):
    # One-time benchmark curation: the requested closed set is digits 1,...,9.
    # This uses labels only to exclude 0, before hiding target training labels.
    indices = torch.where((labels >= 1) & (labels <= 9))[0]
    if limit is not None:
        if limit < 1:
            raise ValueError("The sample limit must be positive or None.")
        generator = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(len(indices), generator=generator)[:limit]]
    return indices


def image_tensor(data, indices):
    images = torch.as_tensor(np.asarray(data))[indices].float().unsqueeze(1) / 255.0
    if images.shape[-2:] != (28, 28):
        images = F.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
    return images


def training_data(root, cfg):
    """Official TRAIN splits only; return source labels and target images."""
    source = MNIST(root, train=True, download=True)
    target = USPS(root, train=True, download=True)
    source_labels = torch.as_tensor(source.targets, dtype=torch.long)
    target_labels = torch.as_tensor(target.targets, dtype=torch.long)
    si = digit_indices(source_labels, cfg.source_samples, cfg.seed)
    ti = digit_indices(target_labels, cfg.target_samples, cfg.seed)
    # Keep native-resolution examples for the input-domain illustration.
    native = (np.asarray(source.data)[si[:8].numpy()],
              np.asarray(target.data)[ti[:8].numpy()])
    source_set = TensorDataset(image_tensor(source.data, si), source_labels[si] - 1)
    target_set = ImagesOnly(image_tensor(target.data, ti))
    return source_set, target_set, native


def test_data(root):
    """Reveal official TEST labels only after both experiments are finished."""
    result = []
    for dataset_type in (MNIST, USPS):
        dataset = dataset_type(root, train=False, download=True)
        labels = torch.as_tensor(dataset.targets, dtype=torch.long)
        indices = digit_indices(labels)
        result.append(TensorDataset(image_tensor(dataset.data, indices), labels[indices] - 1))
    return tuple(result)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, strength):
        ctx.strength = strength
        return z.view_as(z)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),  # Still 7 x 7; no extra pooling.
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 2),
        )
        # Digit class is the pedagogical analogue of structural health state.
        self.health_classifier = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 9))
        self.domain_classifier = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, images):
        z = self.encoder(images)
        return z, self.health_classifier(z)

    def domain_logits(self, z, strength):
        return self.domain_classifier(GradientReversal.apply(z, strength))


def reversal_strength(epoch, step, steps_per_epoch, cfg):
    """Off during warm-up, linear ramp to 60% of training, then constant."""
    ramp_end = 0.6 * cfg.epochs
    if not 0 <= cfg.warmup_epochs < ramp_end:
        raise ValueError("warmup_epochs must be nonnegative and less than 0.6 * epochs.")
    # epoch is zero-based: elapsed=3 means the first three epochs are complete.
    elapsed = epoch + step / steps_per_epoch
    progress = (elapsed - cfg.warmup_epochs) / (ramp_end - cfg.warmup_epochs)
    return cfg.adversarial_weight * min(1.0, max(0.0, progress))


def fit(source_set, target_set, cfg, adaptation=False, device="cpu"):
    """Explicit training entry point. Never accepts target digit labels or test data."""
    if cfg.epochs < 1 or cfg.batch_size < 1 or not len(source_set):
        raise ValueError("Use positive epochs, batch size and source dataset size.")
    if adaptation and (not isinstance(target_set, ImagesOnly) or not len(target_set)):
        raise ValueError("Adaptation requires a nonempty ImagesOnly target dataset.")
    if adaptation:
        reversal_strength(0, 0, 1, cfg)  # Validate the schedule before fitting.
    seed_everything(cfg.seed)  # Identical initialization for both experiments.
    model = DANN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    source_loader = DataLoader(source_set, batch_size=cfg.batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(cfg.seed), num_workers=0)
    if adaptation:
        target_loader = DataLoader(target_set, batch_size=cfg.batch_size, shuffle=True,
                                   generator=torch.Generator().manual_seed(cfg.seed + 1), num_workers=0)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        if adaptation:
            target_iterator = iter(target_loader)
        sums = np.zeros(4)
        for step, (xs, ys) in enumerate(source_loader):
            xs, ys = xs.to(device), ys.to(device)
            zs, logits = model(xs)
            classification = F.cross_entropy(logits, ys)
            domain_loss = xs.new_zeros(())
            domain_accuracy = xs.new_zeros(())
            strength = 0.0
            if adaptation:
                try:
                    xt = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(target_loader)
                    xt = next(target_iterator)
                zt = model.encoder(xt.to(device))
                strength = reversal_strength(epoch, step, len(source_loader), cfg)
                ds = model.domain_logits(zs, strength)
                dt = model.domain_logits(zt, strength)
                # Equal domain weights even when the last batches differ in size.
                domain_loss = 0.5 * (
                    F.cross_entropy(ds, torch.zeros(len(zs), dtype=torch.long, device=device)) +
                    F.cross_entropy(dt, torch.ones(len(zt), dtype=torch.long, device=device)))
                domain_accuracy = 0.5 * ((ds.argmax(1) == 0).float().mean() +
                                         (dt.argmax(1) == 1).float().mean())
            # GRL applies the negative, scaled domain gradient to the encoder only.
            # Do NOT multiply domain_loss by strength again.
            loss = classification + domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += len(xs) * np.array([classification.item(),
                                       domain_loss.item(), domain_accuracy.item(), strength])
        values = sums / len(source_set)
        row = dict(zip(("classification", "domain_loss", "domain_accuracy", "lambda"),
                       values.tolist()))
        history.append(row)
        name = "adapted" if adaptation else "source-only"
        print(f"{name:11s} | epoch {epoch + 1:02d}/{cfg.epochs} | "
              f"digit CE {row['classification']:.3f}" +
              (f" | domain accuracy {row['domain_accuracy']:.1%}" if adaptation else ""))
    return model.eval(), history


def save_checkpoint(path, model, history, cfg, adaptation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 3, "architecture": "DANN",
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "history": history, "config": asdict(cfg), "adaptation": adaptation,
                "digits": list(range(1, 10)), "preprocessing": "gray28_bilinear_0to1"}, path)


def load_checkpoint(path, adaptation, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Set MODE = 'train' and run both training cells, "
                                "or place your trained checkpoint here. No training was started.")
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("format_version") != 3 or saved.get("architecture") != "DANN":
        raise ValueError("This checkpoint is incompatible with the expanded DANN (format 3). "
                         "Train a new pair or load matching format-3 checkpoints.")
    if (saved["adaptation"] != adaptation or
            saved["digits"] != list(range(1, 10)) or saved["preprocessing"] != "gray28_bilinear_0to1"):
        raise ValueError("Checkpoint does not match this experiment / demo format.")
    model = DANN().to(device)
    model.load_state_dict(saved["state_dict"])
    print(f"Loaded {path}; training settings: {saved['config']}")
    return model.eval(), saved["history"], saved["config"]


@torch.no_grad()
def evaluate(model, dataset, device="cpu"):
    model.eval()
    latent, labels, predictions = [], [], []
    for x, y in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0):
        z = model.encoder(x.to(device))
        latent.append(z.cpu())
        labels.append(y)
        predictions.append(model.health_classifier(z).argmax(1).cpu())
    z, y, prediction = (torch.cat(items).numpy() for items in (latent, labels, predictions))
    # Return actual digit labels 1,...,9 for plots and interpretation.
    return {"z": z, "y": y + 1, "prediction": prediction + 1,
            "accuracy": float(np.mean(y == prediction))}


def plot_inputs(native):
    fig, axes = plt.subplots(2, 8, figsize=(10, 3), layout="constrained")
    for row, (images, title) in enumerate(zip(native, ("MNIST: 28 × 28", "USPS: 16 × 16"))):
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(images):
                ax.imshow(images[col], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row, 0].set_title(title, loc="left")
    fig.suptitle("Native inputs: different resolution and handwriting style (unlabelled examples)")
    return fig


def plot_latent(source_result, target_result, title, max_points=1500):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
    rng = np.random.default_rng(7)
    for result, domain, marker in ((source_result, "MNIST", "o"), (target_result, "USPS", "^")):
        indices = rng.choice(len(result["z"]), min(max_points, len(result["z"])), replace=False)
        z, y = result["z"][indices], result["y"][indices]
        axes[0].scatter(*z.T, s=7, alpha=0.35, marker=marker, label=domain)
        ax = axes[1] if domain == "MNIST" else axes[2]
        scatter = ax.scatter(*z.T, c=y, s=7, alpha=0.6, cmap="tab10", vmin=0.5, vmax=9.5)
        ax.set_title(f"{domain}: true digit (evaluation only)")
    axes[0].set_title("Colour / marker = domain")
    axes[0].legend(markerscale=2)
    for ax in axes:
        ax.set_xlabel("$z_1$")
    axes[0].set_ylabel("$z_2$")
    fig.colorbar(scatter, ax=list(axes[1:]), ticks=range(1, 10), label="Digit")
    fig.suptitle(title + " — actual 2D bottleneck, no PCA / t-SNE")
    return fig


def confusion_counts(result):
    """Rows = true digit, columns = predicted digit; inputs use digit labels 1-9."""
    counts = np.zeros((9, 9), dtype=int)
    np.add.at(counts, (result["y"] - 1, result["prediction"] - 1), 1)
    return counts


def plot_confusion_matrices(source_only_result, adapted_result):
    """Compare USPS errors, with each true-digit row expressed as percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, result, title in zip(axes, (source_only_result, adapted_result), ("Source only", "DANN")):
        counts = confusion_counts(result)
        totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(100.0 * counts, totals,
                                out=np.zeros_like(counts, dtype=float), where=totals != 0)
        im = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        for row in range(9):
            for col in range(9):
                if counts[row, col]:
                    ax.text(col, row, f"{percentages[row, col]:.0f}", ha="center", va="center",
                            color="white" if percentages[row, col] > 50 else "black", fontsize=9)
        ax.set_xticks(range(9), range(1, 10))
        ax.set_yticks(range(9), range(1, 10))
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes), label="% of examples of each true digit")
    fig.suptitle("USPS test confusion matrices — labels used only for evaluation")
    return fig
