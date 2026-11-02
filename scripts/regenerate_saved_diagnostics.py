#!/usr/bin/env python3
"""Recreate diagnostic plots and summaries from the published CSV logs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def lookup_table(out: Path) -> None:
    records = []
    for kappa in (2, 5, 10, 20):
        q = (kappa - 1) / (kappa + 1)
        for eps in (5e-3, 1e-3, 5e-4):
            row = {"kappa": kappa, "tolerance": eps}
            for n in (4, 8, 16):
                target = math.log1p(n * eps / (n - 1 - eps))
                t = max(1, math.ceil((1 + math.log(target / math.log(kappa)) / math.log(q)) / 2))
                row[f"n={n}"] = t
            records.append(row)
    with (out / "finite_step_lookup.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def calibration_plots(out: Path) -> None:
    logs = ROOT / "paper-extra-experiments/logs"
    grid = pd.read_csv(logs / "temperature_clipping_grid.csv")
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for beta_r, group in grid.groupby("beta_R"):
        group = group.sort_values("t_max")
        ax.plot(group.t_max, group.measured_eps_max.clip(lower=1e-16), "o-",
                label=rf"measured max $\beta R={beta_r:g}$")
        ax.plot(group.t_max, group.certificate.clip(lower=1e-16), "x--",
                label=rf"cert. $\beta R={beta_r:g}$")
    ax.set(yscale="log", xlabel=r"$t_{\max}$", ylabel="column error / certificate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "temperature_clipping_grid.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    curves = pd.read_csv(logs / "local_predictor_curves.csv")
    for (seed, layer), group in curves.groupby(["seed", "layer"]):
        group = group.sort_values("t")
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        for field, label, style in (
            ("measured_eps_c", "measured", "-"),
            ("kappa_bound", r"$\kappa$-only bound", "--"),
            ("matrix_adaptive_bound", r"$\Delta(A)$ bound", "--"),
            ("local_first_order", "local first-order", "-."),
            ("local_second_order", "second-order", ":"),
        ):
            ax.plot(group.t, group[field].clip(lower=1e-16), style, label=label)
        rho = float(group.rho_star.iloc[0])
        ax.set(yscale="log", xlabel="full Sinkhorn steps t", ylabel=r"$\varepsilon_c^{(2t)}$")
        ax.set_title(rf"Seed {seed}, layer {layer}, $\rho_\ast={rho:.3f}$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"local_predictor_seed{seed}_layer{layer}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("regenerated"))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    import tiny_transformer_relevance_demo as relevance
    import make_paper_plots

    logs = pd.read_csv(ROOT / "relevance-demo/logs/tiny_transformer_relevance_epoch_logs.csv")
    final = logs[logs.epoch == logs.epoch.max()]
    expected = {(seed, policy) for seed in (0, 1, 2) for policy in ("row-only softmax", "Sinkhorn budget")}
    actual = set(zip(final.seed, final.policy))
    if actual != expected or len(final) != len(expected):
        raise ValueError("The final transformer rows must cover both policies and all three seeds.")
    relevance.make_plot(out / "tiny_transformer_relevance.png", logs.to_dict("records"))
    relevance.write_latex_table(out / "tiny_transformer_relevance_table.tex", final.to_dict("records"))
    final.to_csv(out / "tiny_transformer_relevance_summary.csv", index=False)

    subprocess.run([
        sys.executable, str(ROOT / "paper-extra-experiments/regenerate_focused_ablation_3seed.py"),
        str(ROOT / "paper-extra-experiments/logs/focused_ablation_epoch_logs.csv"),
        str(out / "focused"),
    ], check=True)
    make_paper_plots.make_plots(
        str(ROOT / "results-stressed-cap80/logs/epoch_logs.csv"),
        str(out / "stress"), tolerance=5e-4, cap=80,
    )
    calibration_plots(out)
    lookup_table(out)
    print(f"Regenerated saved diagnostics in {out}")


if __name__ == "__main__":
    main()
