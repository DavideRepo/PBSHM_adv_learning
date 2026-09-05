# Your trained DANN checkpoints

No weights are supplied. In the notebook, choose `MODE = "train"` to create:

- `source_only_dann.pt`: source-only model, final epoch.
- `adapted_dann.pt`: adversarial model, final epoch.

Then choose `MODE = "load"`. Loading never starts training. Each file contains model weights, training settings, preprocessing/class metadata and training history. Optimizer state is not saved; these are inference checkpoints, not training-resume snapshots. The loader checks the experiment type, and the notebook checks that the two runs used identical settings.

The files are small and ignored by Git by default. After you train them, you can include your chosen pair in this repository explicitly:

```bash
git add -f checkpoints/source_only_dann.pt checkpoints/adapted_dann.pt
git commit -m "Add trained demonstration checkpoints"
git push
```

Alternatively, distribute the pair separately and place them in this folder. Record your environment and measured results when sharing; no accuracy is implied by the filenames. Only load checkpoints from a trusted source. The loader uses `weights_only=True`.

DANN checkpoints use format version 2. Checkpoints from the earlier autoencoder version are incompatible; train a new matched pair. The new filenames preserve any existing autoencoder checkpoint files.
