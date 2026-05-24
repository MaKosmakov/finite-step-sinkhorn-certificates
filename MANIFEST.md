# Repository Manifest

This repository contains the code, logs, plots, and manuscript source needed to reproduce the finite-step Sinkhorn certificate diagnostics.

## Contents

- `manuscript/`: manuscript source and compiled PDF.
- `code/`: CPU/GPU training and diagnostic scripts.
- `notebooks/`: Kaggle-ready notebooks matching the paper diagnostics.
- `logs/`: raw CSV/JSON/TeX tables used by the manuscript.
- `plots/`: regenerated paper-facing diagnostic figures.
- `README.md`, `RUN_GPU_COMMANDS.md`: command and environment overview.

## Notes

- CIFAR-10 data is not bundled. The scripts expect the standard CIFAR-10 Python archive or an extracted CIFAR-10 directory.
- No local build scratch files, raw datasets, or external git metadata are included.
