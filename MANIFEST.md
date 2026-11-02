# Reproduction map

Run commands from the repository root. `RUN_GPU_COMMANDS.md` gives training
commands; the following map identifies the saved evidence for each paper item.

| Paper item | Saved evidence | Regeneration |
| --- | --- | --- |
| Finite-step lookup | Calibration formula in the manuscript | `scripts/regenerate_saved_diagnostics.py` |
| Matched transformer table and Figure 1 | `relevance-demo/logs/` | `scripts/regenerate_saved_diagnostics.py` |
| Certificate comparison and Dobrushin audit | `relevance-demo/mixers/` | `scripts/dobrushin_diagnostic.py` |
| Four-policy controller ablation | `paper-extra-experiments/logs/focused_ablation_epoch_logs.csv` | `scripts/regenerate_saved_diagnostics.py` |
| Post-hoc budget sweep | Saved final row-softmax mixer stacks | `scripts/budget_sensitivity_diagnostic.py` |
| Temperature-clipping and local-predictor figures | `paper-extra-experiments/logs/` | `scripts/regenerate_saved_diagnostics.py` |
| Stress-run audit plots | `results-stressed-cap80/logs/epoch_logs.csv` | `code/make_paper_plots.py` |

The CSV files store the reported measurements. Regenerating figures does not
repeat training; rerunning the budget diagnostic produces new hardware-dependent
timings. GPU training runs are stochastic and need not reproduce every logged
floating-point value exactly.
