# Finite-Step Sinkhorn Certificates

Code and saved diagnostics for **Finite-Step Sinkhorn Certificates for Nonnegative
Stream-Mixing Products**, by Maksim Kosmakov, accepted in *Transactions on Machine
Learning Research* (2026).

[Paper and reviews](https://openreview.net/forum?id=NyEeOl6Gn7)

The code measures how finite Sinkhorn normalization controls the backward gain of
a stack of nonnegative stream mixers. The reported experiments use CIFAR-10:
a matched small-transformer run, a four-policy controller ablation, and calibration
and local-predictor diagnostics.

## Install

Use Python 3.10 or later. Install PyTorch appropriate for your CPU or CUDA device,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Check the saved results on CPU

No dataset or training is needed for these commands:

```bash
python scripts/dobrushin_diagnostic.py --roots relevance-demo/mixers \
  --csv regenerated/dobrushin_diagnostic.csv \
  --tex regenerated/dobrushin_diagnostic_table.tex \
  --comparison-tex regenerated/certificate_comparison_table.tex
python scripts/budget_sensitivity_diagnostic.py \
  --csv regenerated/budget_sensitivity.csv \
  --tex regenerated/budget_sensitivity_table.tex
python scripts/regenerate_saved_diagnostics.py --out-dir regenerated
```

The first command audits the six saved final mixer stacks: two policies, three
seeds, 24 layers, and eight streams. The budget sweep re-normalizes the saved
row-softmax stacks under different budgets. Its CPU timings will vary by machine.
The last command recreates the figures and summaries from the CSV logs and checks
the finite-step lookup values. Checked-in outputs retain the measurements used in
the published paper.

## Run the experiments

See [RUN_GPU_COMMANDS.md](RUN_GPU_COMMANDS.md) for the exact training commands,
a real-data CPU smoke test, and the settings for each reported experiment.
The focused controller comparison uses seeds `0,1,2` for five epochs; the matched
transformer run uses the same three seeds, 5,000 training images, 2,000 test images,
and eight epochs. The transformer diagnostic uses a prescribed mixer initialization
to expose the gain difference between the two normalization policies.

Kaggle notebooks are in `notebooks/`. They contain the helper modules needed for
execution. Enable a GPU and add the
[cifar10-python dataset](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python).
The dataset is not included in this repository. On Kaggle it is mounted at
`/kaggle/input/cifar10-python`; locally, point the scripts to the extracted
`cifar-10-batches-py` directory or its parent, as documented in the commands.

## Files

- `code/`: training, normalization, and plotting code. The extra-experiment runner
  imports `code/adaptive_budget_ablation_kaggle.py`.
- `scripts/`: post-hoc certificate audits and regeneration from saved results.
- `relevance-demo/`: transformer configuration, epoch logs, final mixer stacks,
  and figures.
- `paper-extra-experiments/`: focused ablation logs, calibration grid, and local
  predictor curves. The saved local-predictor curves cover seeds 0 and 1.
- `results-stressed-cap80/`: the cap-80 stress-run logs and audit figures.
- `results/`, `tables/`: post-hoc certificate and budget-sensitivity outputs.
- `manuscript/`: camera-ready LaTeX, PDF, styles, and required figure/table files.

## AI assistance

ChatGPT and OpenAI Codex assisted with code development and debugging, experiment
and figure scripts, mathematical checks, and manuscript editing. The author takes responsibility for
the paper and the accompanying code.

## Citation and license

Use the citation in `CITATION.cff`, or cite the paper through its OpenReview page.
The code is released under the MIT License. The manuscript is distributed under
CC BY 4.0; third-party style files retain their own notices.
