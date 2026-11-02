# Experiment commands

Run these commands from the repository root. Install `requirements.txt` first.
For Kaggle, enable a GPU and attach
[cifar10-python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python).
Training outputs go to fresh directories so the published logs stay intact.

## Matched transformer diagnostic

```bash
python code/tiny_transformer_relevance_demo.py \
  --data_dir /kaggle/input/cifar10-python \
  --out_dir rerun/relevance-demo \
  --seeds 0 1 2 --epochs 8 --train_size 5000 --test_size 2000 \
  --depth 24 --n_streams 8 --dim 24 --heads 2 --patch 8 --batch_size 128 \
  --beta 4 --clip_logits 8 --mixer_init_noise 0.05 --collapse_strength 5 \
  --lr 0.002 --weight_decay 0.0001 --grad_clip 0 \
  --sinkhorn_t_max 40 --sinkhorn_eps 5e-4 --save_mixers
```

The same seed and minibatch order are used for both mixer policies. The prescribed
column-concentrating initialization is part of this diagnostic. The saved final
stacks have shape `(24,8,8)` and are evaluated independently of the nonlinear
branch.

## Four-policy controller ablation

This uses depth 128, eight streams, width 128, cap 80, tolerance `5e-4`, and
seeds `0,1,2` for five epochs. The helper module is included in `code/`.

```bash
python code/paper_extra_experiments_3seeds_5epochs_kaggle.py
```

On Kaggle the output is
`/kaggle/working/paper-extra-experiments-3seeds-5epochs/`.
For local execution, set the `data_root` and `out_dir` entries in the runner's
configuration. Pass the parent of `cifar-10-batches-py` as `data_root`.

## Temperature-clipping and trained local predictors

```bash
python code/paper_extra_experiments_kaggle.py
```

This runs the four policies and then exports the calibration grid and local
predictors from the trained adaptive model. The calibration grid uses 64 random
clipped-logit matrices per cell in float64; it is a numerical design-rule check.
The saved local-predictor curves in the release cover seeds 0 and 1, layers
0, 64, and 127. The focused five-epoch policy table uses three completed seeds.

## Cap-80 stress diagnostic

```bash
python code/train_cifar10_mhc.py \
  --profile stress_full --epochs 5 --seeds 0 1 2 \
  --batch_size 256 --num_workers 4 \
  --data_root /kaggle/input/cifar10-python \
  --out_dir rerun/results-stressed-cap80 --amp
```

Use `--data_parallel` for multiple GPUs. This command uses the `stress_full`
profile's cap of 80; it is separate from the cap-40 matched transformer run.

## CPU smoke test on real data

Replace the data path with an extracted CIFAR-10 directory or its parent.
This checks one training and evaluation batch; it is not a reported result.

```bash
python code/train_cifar10_mhc.py \
  --profile baseline --epochs 1 --seeds 0 \
  --num_workers 0 --batch_size 64 --n_streams 4 --d 32 --depth 4 \
  --sinkhorn_t_max 8 --sinkhorn_eps_tol 5e-4 \
  --max_train_batches 1 --max_eval_batches 1 \
  --data_root /kaggle/input/cifar10-python --out_dir results-cpu-smoke
```

## Regenerate from saved results

```bash
python scripts/regenerate_saved_diagnostics.py --out-dir regenerated
python scripts/dobrushin_diagnostic.py --roots relevance-demo/mixers \
  --csv regenerated/dobrushin_diagnostic.csv \
  --tex regenerated/dobrushin_diagnostic_table.tex \
  --comparison-tex regenerated/certificate_comparison_table.tex
python scripts/budget_sensitivity_diagnostic.py \
  --csv regenerated/budget_sensitivity.csv \
  --tex regenerated/budget_sensitivity_table.tex
```

The regeneration script reads the checked-in CSV files, checks that all three
focused seeds completed five epochs, and recreates the figures, summaries, and
lookup table without training. Budget-sweep timings are measured again on the
current CPU; the paper's original times remain in `results/budget_sensitivity.csv`.

## Compile the manuscript

```bash
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error sinkhorn_tmlr_final.tex
pdflatex -interaction=nonstopmode -halt-on-error sinkhorn_tmlr_final.tex
pdflatex -interaction=nonstopmode -halt-on-error sinkhorn_tmlr_final.tex
```

The source includes the required tables, figures, and official TMLR style file.
