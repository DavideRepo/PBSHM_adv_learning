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

1. Set `MODE = "train"` and run the notebook to train both experiments yourself. This saves `checkpoints/source_only_dann_v3.pt` and `checkpoints/adapted_dann_v3.pt`.
2. Set `MODE = "load"` to reuse your saved models and generate the evaluation plots without training.

**No models have been trained and no pretrained weights are supplied.** Loading is the default; missing checkpoints give a clear error instead of starting training. Training again overwrites the two DANN checkpoint files. See [checkpoints/README.md](checkpoints/README.md) for sharing your models.

The demonstration uses NumPy, PyTorch, torchvision and Matplotlib; JupyterLab runs the notebook. For a CPU-specific PyTorch installation, follow the [official installer](https://pytorch.org/get-started/locally/) before installing the requirements.

Default settings: CPU, up to four CPU threads, 24,000 MNIST training examples, all eligible USPS training examples, batch size 128, 30 epochs and three adversarial warm-up epochs. Set `source_samples=None` in `Config` to use all eligible source training images. Set `EPOCHS` in the notebook setup cell to change the duration. Adam uses a constant learning rate of `1e-3` throughout; no learning-rate decay is applied. Runtime and accuracy of this revision have not been benchmarked.

## USPS download certificate error

If section 2 stops at `USPS(..., download=True)` with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`, Python could not build a trusted certificate chain for the USPS download host. The traceback alone does not identify whether the cause is the server chain, local certificate configuration or a network proxy.

You can use your browser to download the original files instead:

| Split | Original archive | Save as, with the default `DATA_DIR` |
|---|---|---|
| Training | [usps.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2) | `data/usps.bz2` |
| Test | [usps.t.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2) | `data/usps.t.bz2` |

1. Download both files through a browser that accepts the HTTPS connection normally. If the browser also reports a certificate warning, do not bypass it; the certificate or network configuration needs attention.
2. Save the archives directly inside the notebook's `DATA_DIR`, keeping their exact names and leaving them compressed. Do not put them inside `data/USPS/`.
3. Rerun the failed cell. Torchvision uses the local archives and skips their downloads, including the test archive later in the notebook. This does not load test labels during training.


## Model and objective

The encoder uses two convolution/ReLU/pooling blocks, then a third `3×3` convolution (`32 → 32`) with ReLU on the 7×7 feature maps, followed by `FC(64) → FC(2)`. Each classifier has two 32-unit hidden layers with ReLU. The complete model has **117,261 parameters**.

| Component | Objective it minimizes |
|---|---|
| Encoder | `digit_CE - lambda * domain_CE` |
| Digit / health classifier | MNIST `digit_CE` |
| Domain classifier | `domain_CE`, weighted equally across MNIST and USPS |

The source-only experiment omits the domain loss. DANN uses `loss = classification + domain_loss`; gradient reversal applies the negative sign and `lambda` only on the path back to the encoder. At inference, digit prediction uses only the encoder and digit classifier.

Both runs start from identical weights and receive identical MNIST batches and update counts. In DANN, adversarial pressure is zero during warm-up, rises linearly to `adversarial_weight=0.1` at 60% of training, and stays constant thereafter (for 30 epochs: warm-up ends after epoch 3, ramp ends after epoch 18); the domain classifier can learn during warm-up while its encoder gradient is zero. Target batches are reshuffled and reused when exhausted.

## Data and evaluation

- The shared class set is **digits 1–9**. Dataset metadata is used to remove zero (a stated closed-set curation assumption) and to arrange the native-input illustration into five random examples per digit. This illustration does not supply target labels to training. Classifier indices are 0–8, converted back to digit labels for plotting.
- MNIST stays at 28 × 28; USPS is bilinearly resized from 16 × 16. Both are scaled to [0, 1].
- Source-only training uses MNIST training images and labels. DANN also uses USPS training **images only**: the target wrapper stores no digit labels.
- Knowing the domain (MNIST or USPS) is different from knowing the digit.
- Official test images and labels are opened after both experiments finish. Training saves the final epoch, with no USPS-based early stopping or checkpoint selection.

The hoped-for pattern is lower USPS accuracy before adaptation and an improvement afterwards. Neither outcome is guaranteed, particularly with a 2D bottleneck. Domain mixing can align different classes incorrectly. Using USPS scores to select settings or seeds would make those labels validation data rather than an untouched final evaluation.

In the PBSHM analogy, digits represent shared health/damage classes and the datasets represent different structures or populations. This image example does not establish performance on structural measurements.

## Checks without training

```bash
python -m unittest discover -s tests -v
```

Offline checks use synthetic tensors for model shapes, gradient reversal, label handling, checkpoint compatibility and notebook/plot execution. They never download datasets, call `fit`, or update model parameters. They do not test convergence or transfer accuracy.

## References

- Ganin et al. (2016), [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/v17/15-239.html).
- Torchvision [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and [USPS](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.USPS.html).
