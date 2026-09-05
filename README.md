# PBSHM: adversarial domain-invariant autoencoder

A short teaching demonstration of **MNIST → USPS transfer**, using an autoencoder with a **two-dimensional latent space**, a digit (health-state) classifier, and a gradient-reversal domain classifier.

Start with **[adversarial_autoencoder.ipynb](adversarial_autoencoder.ipynb)**. The implementation is in [demo.py](demo.py).

## Run

Use Python 3.10 or newer. In a virtual environment, install:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open the notebook using that environment's Python kernel. The demonstration uses only NumPy, PyTorch, torchvision and Matplotlib; JupyterLab is included to run the notebook. For a CPU-specific PyTorch installation, use the command provided by the [official PyTorch installer](https://pytorch.org/get-started/locally/) before installing the requirements.

1. Set `MODE = "train"` in the first code cell and run the notebook to train both experiments yourself. This writes `checkpoints/source_only.pt` and `checkpoints/adapted.pt`.
2. Set `MODE = "load"` for subsequent demonstrations. This loads your saved models and histories, then produces the evaluation plots without training.

**No training has been performed and no pretrained weights are supplied.** The default is `"load"`; a missing checkpoint gives a clear error instead of starting training. Training again overwrites the two named checkpoint files. Retain a matched pair with identical settings. See [checkpoints/README.md](checkpoints/README.md) for sharing your weights.

Default settings: CPU, at most four CPU threads in the notebook, 12,000 MNIST training examples, all eligible USPS training examples, batch size 128, 20 epochs, and a three-epoch adversarial warm-up. Runtime and accuracy have not been benchmarked. The network has about 215,000 parameters; the bottleneck and all heads are deliberately small.

## Notebook sequence

1. Compare unlabelled input images at their native resolutions.
2. Explain the encoder, decoder, digit classifier, domain classifier and gradient reversal.
3. Train or load a source-only autoencoder/classifier.
4. Train or load the same architecture with adversarial domain adaptation.
5. Inspect training losses and balanced domain accuracy.
6. Reveal held-out test labels; compare source-only and adapted accuracies and two-dimensional latent plots.
7. Inspect reconstructions and discuss the PBSHM analogy and limitations.

## Experimental boundary

| Data | Source-only training | Adapted training | Final evaluation |
|---|---|---|---|
| MNIST training images and digit labels | Classification + reconstruction | Classification + reconstruction + domain discrimination | — |
| USPS training images | Unused | Domain discrimination only | — |
| USPS training digit labels | Unused after class filtering | Unavailable to the training interface | — |
| MNIST / USPS official test splits | Unused | Unused | Digit accuracy and plots |

The class set is **1–9**, excluding zero. Defining this closed-set benchmark uses metadata once to remove zero; this is stated explicitly because it requires knowing class membership during curation. Afterwards the USPS training dataset stores only images. The adaptation loop gets domain labels (which dataset), never USPS digit labels. No test data is loaded until both training/loading stages finish. Training uses a fixed epoch budget and saves the final epoch; no USPS-based selection or early stopping.

Both experiments start from identical weights with identical source data order and update counts. Both reconstruct **source images only**; the sole added objective is domain discrimination with its reversed encoder gradient. This also avoids confounding adversarial alignment with added target reconstruction. The domain loss weights the two domains equally despite different dataset sizes; target batches are recycled with fresh shuffling when exhausted.

MNIST stays at 28 × 28; USPS is bilinearly resized from 16 × 16. Both use grayscale values in [0, 1]. Digit labels 1–9 are mapped to classifier indices 0–8 and mapped back for plots.

## Learning objective

The encoder minimizes `digit_CE + beta * source_reconstruction_MSE - lambda * domain_CE`. The domain classifier minimizes `domain_CE`. A gradient-reversal layer handles the negative sign and `lambda` **only on the path to the encoder**; the actual scalar loss adds the domain cross-entropy normally. The domain head is not used to predict digits at inference.

This is a DANN-style autoencoder, not an adversarial autoencoder that matches its latent distribution to a prior. Digit identity is analogous to a shared health/damage state, and the two image datasets are analogous to different structures or populations. This is a pedagogical analogy, not evidence of performance on structural data.

The intended outcome is a source–target accuracy gap followed by improved USPS accuracy with adaptation. **Neither result is guaranteed**, especially with a 2D bottleneck. The notebook plots actual results and reports signed accuracy changes. Domain confusion alone does not establish correct class alignment or successful transfer. Inspecting USPS scores to choose settings would make those labels validation data; it would no longer be a strictly untouched final evaluation.

## Checks without training

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic tensors: model shapes, gradient-reversal direction, label handling, checkpoint round-trip, and plotting/notebook smoke checks. They never download datasets, call `fit`, or update model parameters. They do not establish convergence or transfer performance.

## References

- Ganin et al. (2016), [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/v17/15-239.html).
- Torchvision [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and [USPS](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html) datasets.
