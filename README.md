# Finite-Step Sinkhorn Certificates

This repository accompanies the paper **Finite-Step Sinkhorn Certificates for Nonnegative Stream-Mixing Products**.

The project studies a simple practical question: if a neural module uses only a finite number of Sinkhorn normalization steps, what stability budget did those steps actually certify? The code here reproduces the CIFAR-10 diagnostic experiments, adaptive-budget ablations, temperature-clipping checks, and local-predictor plots from the paper.

The experiments are diagnostic rather than benchmark claims. They are meant to make the certificate quantities visible: column-marginal error, product certificates, exact propagated backward gain, cap failures, Sinkhorn step counts, and local spectral rates.

## What is in this repository

- `code/train_cifar10_mhc.py`  
  Main CIFAR-10 stress runner. It trains paired softmax/Sinkhorn variants and records the certificate diagnostics used in the paper.

- `code/paper_extra_experiments_kaggle.py`  
  Extra diagnostics: temperature-clipping grid and local predictor curves.

- `code/paper_extra_experiments_3seeds_5epochs_kaggle.py`  
  Focused controller ablation over seeds `0,1,2`: fixed cap, uniform tolerance, adaptive log-budget, and certificate-aware penalty.

- `code/make_paper_plots.py`  
  Regenerates the cleaner paper-facing plots from CSV logs.

- `notebooks/`  
  Kaggle-ready notebooks matching the script versions.

- `logs/` and `plots/`  
  Raw logs and generated figures used by the current paper draft.

- `manuscript/`  
  A copy of the current manuscript source/PDF for convenience.

## Quick start

Run a tiny CPU smoke test first:

```bash
python code/train_cifar10_mhc.py \
  --profile baseline \
  --epochs 1 --seeds 0 \
  --num_workers 0 --batch_size 64 \
  --n_streams 4 --d 32 --depth 4 \
  --sinkhorn_t_max 8 --sinkhorn_eps_tol 5e-4 \
  --max_train_batches 1 --max_eval_batches 1 \
  --data_root /path/to/cifar10 \
  --out_dir results-cpu-smoke
```

For the full GPU stress diagnostics used in the paper:

```bash
python code/train_cifar10_mhc.py \
  --profile stress_full \
  --epochs 5 --seeds 0 1 2 \
  --batch_size 256 --num_workers 4 \
  --data_root /path/to/cifar10 \
  --out_dir results-stressed-cap80 \
  --amp
```

Then regenerate paper-style plots:

```bash
python code/make_paper_plots.py \
  --csv results-stressed-cap80/logs/epoch_logs.csv \
  --out_dir results-stressed-cap80/plots
```

Run the extra diagnostics:

```bash
python code/paper_extra_experiments_kaggle.py
python code/paper_extra_experiments_3seeds_5epochs_kaggle.py
```

## Data

CIFAR-10 is not bundled. Use the standard CIFAR-10 Python archive or an extracted CIFAR-10 directory and pass it through `--data_root` or the corresponding notebook variable.

## Reproducibility notes

The paper tables and figures are regenerated from the checked-in CSV/JSON logs and scripts. The reported focused ablation uses seeds `0,1,2`. Larger benchmark claims are intentionally out of scope; the goal is to inspect whether the certificate quantities behave as predicted by the theory.

## License

This code release is distributed under the MIT License. See `LICENSE`.
