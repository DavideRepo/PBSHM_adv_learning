"""Offline, synthetic checks only. No fit calls, optimizer steps or datasets downloaded."""
import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

import demo


class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_shapes(self):
        model = demo.DANN().eval()
        with torch.no_grad():
            z, logits = model(torch.rand(3, 1, 28, 28))
            domains = model.domain_logits(z, 0.1)
        self.assertEqual(tuple(z.shape), (3, 2))
        self.assertEqual(tuple(logits.shape), (3, 9))
        self.assertEqual(tuple(domains.shape), (3, 2))

    def test_grl_reverses_encoder_gradient_only(self):
        head = torch.nn.Linear(2, 2)
        z = torch.tensor([[0.2, 0.8]], requires_grad=True)
        plain = torch.autograd.grad(head(z).square().sum(), (z, head.weight))
        reversed_ = torch.autograd.grad(head(demo.GradientReversal.apply(z, 0.3)).square().sum(),
                                        (z, head.weight))
        torch.testing.assert_close(reversed_[0], -0.3 * plain[0])
        torch.testing.assert_close(reversed_[1], plain[1])
        zero = torch.autograd.grad(demo.GradientReversal.apply(z, 0).sum(), z)[0]
        torch.testing.assert_close(zero, torch.zeros_like(z))

    def test_curation_and_hidden_target_labels(self):
        # Stub native dataset construction to verify the whole preparation boundary offline.
        class FakeMNIST:
            def __init__(self, root, train, download):
                self.targets = torch.arange(10)
                self.data = torch.arange(10, dtype=torch.uint8)[:, None, None].expand(10, 28, 28)

        class FakeUSPS:
            def __init__(self, root, train, download):
                self.targets = list(range(10))
                self.data = np.arange(10, dtype=np.uint8)[:, None, None] * np.ones((10, 16, 16), dtype=np.uint8)

        with patch.object(demo, "MNIST", FakeMNIST), patch.object(demo, "USPS", FakeUSPS):
            source, target, native = demo.training_data("unused", demo.Config(source_samples=None))
        self.assertEqual(len(source), 9)
        self.assertEqual(source.tensors[1].tolist(), list(range(9)))
        self.assertEqual(set(vars(target)), {"images"})
        self.assertEqual(tuple(target[0].shape), (1, 28, 28))
        torch.testing.assert_close(target[0], torch.full((1, 28, 28), 1 / 255))
        self.assertEqual(native[1].shape[-2:], (16, 16))
        indices = demo.digit_indices(torch.arange(10), limit=5, seed=7)
        self.assertEqual(len(indices), 5)
        self.assertNotIn(0, indices.tolist())

    def test_checkpoint_round_trip_and_missing_file(self):
        model = demo.DANN().eval()  # Random initialization; never fitted.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            demo.save_checkpoint(path, model, [], demo.Config(), adaptation=False)
            loaded, history, config = demo.load_checkpoint(path, adaptation=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
            self.assertEqual(history, [])
            self.assertEqual(config["seed"], 7)
            with self.assertRaises(ValueError):
                demo.load_checkpoint(path, adaptation=True)
            with self.assertRaisesRegex(FileNotFoundError, "No training was started"):
                demo.load_checkpoint(Path(folder) / "missing.pt", adaptation=False)
            # Earlier architectures must fail before loading model weights.
            for version in (1, 2):
                torch.save({"format_version": version, "architecture": "DANN"}, path)
                with self.assertRaisesRegex(ValueError, "expanded DANN"):
                    demo.load_checkpoint(path, adaptation=False)

    def test_linear_adversarial_schedule(self):
        cfg = demo.Config()
        for epoch in (0, 1, 2, 3):
            self.assertEqual(demo.reversal_strength(epoch, 0, 100, cfg), 0.0)
        self.assertAlmostEqual(demo.reversal_strength(10, 50, 100, cfg), 0.05)
        for epoch in (18, 24, 29, 30):
            self.assertEqual(demo.reversal_strength(epoch, 0, 100, cfg), 0.1)
        # Changing the epoch budget moves the endpoint to 60% of that budget.
        longer = demo.Config(epochs=50)
        self.assertAlmostEqual(demo.reversal_strength(16, 50, 100, longer), 0.05)
        self.assertEqual(demo.reversal_strength(30, 0, 100, longer), 0.1)
        with self.assertRaisesRegex(ValueError, "0.6"):
            demo.reversal_strength(0, 0, 1, demo.Config(epochs=5))

    def test_confusion_matrix_orientation_and_normalization(self):
        result = {"y": np.array([1, 1, 1, 2, 9]),
                  "prediction": np.array([1, 1, 2, 1, 9])}
        counts = demo.confusion_counts(result)
        self.assertEqual(counts.sum(), 5)
        self.assertEqual(counts[0, 0], 2)
        self.assertEqual(counts[0, 1], 1)
        self.assertEqual(counts[1, 0], 1)
        self.assertEqual(counts[8, 8], 1)
        fig = demo.plot_confusion_matrices(result, result)
        percentages = fig.axes[0].images[0].get_array()
        self.assertAlmostEqual(percentages[0, 0], 200 / 3)
        self.assertEqual(percentages[1, 0], 100)
        self.assertEqual(percentages[8, 8], 100)
        self.assertTrue(np.isfinite(percentages).all())  # Absent digits stay zero.
        np.testing.assert_allclose(percentages[2:8], 0)
        fig.canvas.draw()
        plt.close(fig)

    def test_notebook_and_plots_without_training(self):
        path = Path(__file__).resolve().parents[1] / "dann.ipynb"
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        dataset = TensorDataset(torch.rand(9, 1, 28, 28), torch.arange(9))
        model = demo.DANN().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = demo.evaluate(model, dataset)
        self.assertEqual(result["y"].tolist(), list(range(1, 10)))
        self.assertTrue(0 <= result["accuracy"] <= 1)
        figures = [demo.plot_inputs((np.zeros((8, 28, 28)), np.zeros((8, 16, 16)))),
                   demo.plot_latent(result, result, "Synthetic check")]
        for figure in figures:
            figure.canvas.draw()
            plt.close(figure)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_notebook_load_workflow_without_training(self):
        path = Path(__file__).resolve().parents[1] / "dann.ipynb"
        notebook = json.loads(path.read_text())
        images = torch.rand(18, 1, 28, 28)
        source = TensorDataset(images, torch.arange(18) % 9)
        target = demo.ImagesOnly(images)
        native = (np.zeros((8, 28, 28)), np.zeros((8, 16, 16)))
        history = [{"classification": 2.0, "domain_loss": 0.7,
                    "domain_accuracy": 0.5, "lambda": 0.0}]
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for name, adaptation in (("source_only_dann_v3", False), ("adapted_dann_v3", True)):
                    demo.save_checkpoint(Path("checkpoints") / f"{name}.pt", demo.DANN(),
                                         history, demo.Config(), adaptation)
                with patch.object(demo, "training_data", return_value=(source, target, native)), \
                     patch.object(demo, "test_data", return_value=(source, source)), \
                     patch.object(demo, "fit", side_effect=AssertionError("Training prohibited")), \
                     patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No updates")), \
                     patch.object(plt, "show"), contextlib.redirect_stdout(io.StringIO()):
                    scope = {}
                    for cell in notebook["cells"]:
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), str(path), "exec"), scope)
                    for number in plt.get_fignums():
                        plt.figure(number).canvas.draw()
            finally:
                plt.close("all")
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
