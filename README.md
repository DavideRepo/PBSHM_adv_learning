# PBSHM: domain-adversarial learning

A short teaching demonstration of **MNIST → USPS transfer** using a plain **domain-adversarial neural network (DANN)**: a small convolutional encoder, a **2D latent space**, a digit/health classifier, and a domain classifier with gradient reversal.

Start with **[dann.ipynb](dann.ipynb)**. The implementation is in [demo.py](demo.py).

## Run

Use Python 3.10 or newer. In a virtual environment:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `dann.ipynb` using that environment's Python kernel.

1. Set `MODE = "train"` and run the notebook to train both experiments yourself. This saves `checkpoints/source_only_dann.pt` and `checkpoints/adapted_dann.pt`.
2. Set `MODE = "load"` to reuse your saved models and generate the evaluation plots without training.

**No models have been trained and no pretrained weights are supplied.** Loading is the default; missing checkpoints give a clear error instead of starting training. Training again overwrites the two DANN checkpoint files. See [checkpoints/README.md](checkpoints/README.md) for sharing your models.

The demonstration uses NumPy, PyTorch, torchvision and Matplotlib; JupyterLab runs the notebook. For a CPU-specific PyTorch installation, follow the [official installer](https://pytorch.org/get-started/locally/) before installing the requirements.

Default settings: CPU, up to four CPU threads, 12,000 MNIST training examples, all eligible USPS training examples, batch size 128, 20 epochs and three adversarial warm-up epochs. Set `source_samples=None` in `Config` to use all eligible source training images. These settings have not been tuned or benchmarked.

## What students see

1. Input differences between MNIST and USPS.
2. A simple explanation of the encoder, two classifiers and gradient reversal.
3. Source-only training and DANN adaptation with matched initialization and training budgets.
4. Classification loss and domain-accuracy curves.
5. Held-out digit accuracies and actual 2D latent plots coloured by domain and digit.
6. Discussion questions connecting transfer learning to PBSHM.

## Model and objective

The encoder uses two convolution/ReLU/pooling blocks followed by `FC(64) → FC(2)`. Each classifier has one 32-unit hidden layer. The complete model has **105,901 parameters**.

| Component | Objective it minimizes |
|---|---|
| Encoder | `digit_CE - lambda * domain_CE` |
| Digit / health classifier | MNIST `digit_CE` |
| Domain classifier | `domain_CE`, weighted equally across MNIST and USPS |

The source-only experiment omits the domain loss. DANN uses `loss = classification + domain_loss`; gradient reversal applies the negative sign and `lambda` only on the path back to the encoder. At inference, digit prediction uses only the encoder and digit classifier.

Both runs start from identical weights and receive identical MNIST batches and update counts. In DANN, adversarial pressure ramps up after the warm-up; the domain classifier can learn during warm-up while its encoder gradient is zero. Target batches are reshuffled and reused when exhausted.

## Data and evaluation

- The shared class set is **digits 1–9**. Dataset metadata is used once to remove zero; this is a stated closed-set curation assumption. Classifier indices are 0–8, converted back to digit labels for plotting.
- MNIST stays at 28 × 28; USPS is bilinearly resized from 16 × 16. Both are scaled to [0, 1].
- Source-only training uses MNIST training images and labels. DANN also uses USPS training **images only**: the target wrapper stores no digit labels.
- Knowing the domain (MNIST or USPS) is different from knowing the digit.
- Official test images and labels are opened after both experiments finish. Training saves the final epoch, with no USPS-based early stopping or checkpoint selection.

The hoped-for pattern is lower USPS accuracy before adaptation and an improvement afterwards. Neither outcome is guaranteed, particularly with a 2D bottleneck. Domain mixing can align different classes incorrectly. Using USPS scores to select settings or seeds would make those labels validation data rather than an untouched final evaluation.

In the PBSHM analogy, digits represent shared health/damage classes and the datasets represent different structures or populations. This image example does not establish performance on structural measurements.

## Changes from the earlier version

This plain DANN replaces the autoencoder: the decoder, reconstruction loss and reconstruction plots are removed. The notebook is now `dann.ipynb`. DANN checkpoints use new filenames and format version 2; old autoencoder checkpoints are not compatible and are not overwritten by the new defaults.

## Checks without training

```bash
python -m unittest discover -s tests -v
```

Offline checks use synthetic tensors for model shapes, gradient reversal, label handling, checkpoint compatibility and notebook/plot execution. They never download datasets, call `fit`, or update model parameters. They do not test convergence or transfer accuracy.

## References

- Ganin et al. (2016), [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/v17/15-239.html).
- Torchvision [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and [USPS](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html).
