# Your trained DANN checkpoints

No weights are supplied. In the notebook, choose `MODE = "train"` to create:

- `source_only_dann_v3.pt`: source-only model, final epoch.
- `adapted_dann_v3.pt`: adversarial model, final epoch.

Then choose `MODE = "load"`. Loading never starts training. Each file contains model weights, training settings, preprocessing/class metadata and training history. Optimizer state is not saved; these are inference checkpoints, not training-resume snapshots. The loader checks the experiment type, and the notebook checks that the two runs used identical settings.

The files are small and ignored by Git by default. After you train them, you can include your chosen pair in this repository explicitly:

```bash
git add -f checkpoints/source_only_dann_v3.pt checkpoints/adapted_dann_v3.pt
git commit -m "Add trained demonstration checkpoints"
git push
```

Alternatively, distribute the pair separately and place them in this folder. Record your environment and measured results when sharing; no accuracy is implied by the filenames. Only load checkpoints from a trusted source. The loader uses `weights_only=True`.

The expanded DANN uses format version 3. Earlier DANN (format 2) and autoencoder (format 1) checkpoints are incompatible. After pulling this version, restart your notebook kernel, set `MODE = "train"` and create a new matched pair. The `_v3.pt` filenames preserve your earlier checkpoints. Training again overwrites only the two version-3 files.
