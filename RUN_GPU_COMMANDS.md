# Reproduction Commands

These commands reproduce the main diagnostics from the paper. Paths are intentionally written as placeholders so the commands can be run on a local machine, Kaggle, or another GPU environment.

## CPU smoke test

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

## Full CIFAR-10 stress run

```bash
python code/train_cifar10_mhc.py \
  --profile stress_full \
  --epochs 5 --seeds 0 1 2 \
  --batch_size 256 --num_workers 4 \
  --data_root /path/to/cifar10 \
  --out_dir results-stressed-cap80 \
  --amp
```

## Regenerate plots

```bash
python code/make_paper_plots.py \
  --csv results-stressed-cap80/logs/epoch_logs.csv \
  --out_dir results-stressed-cap80/plots
```

## Extra diagnostics

```bash
python code/paper_extra_experiments_kaggle.py
python code/paper_extra_experiments_3seeds_5epochs_kaggle.py
```

The extra diagnostics produce the focused controller-ablation table, temperature-clipping grid, and local-predictor curve used in the paper.
