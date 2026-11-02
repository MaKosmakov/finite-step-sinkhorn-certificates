"""Regenerate the paper-facing focused-ablation table and plots.

The current manuscript reports the fixed seed set 0,1,2.  This script filters
the raw epoch log to those seeds and rewrites the table, summary CSV, and
focused plots used by the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


POLICIES = ["fixed", "uniform", "adaptive", "cert_penalty"]
LABELS = {
    "fixed": r"Fixed $t_{\max}=80$",
    "uniform": "Uniform tolerance",
    "adaptive": "Adaptive log-budget",
    "cert_penalty": "Certificate-aware penalty",
}
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_root", type=Path, default=Path("paper-extra-experiments"))
    args = parser.parse_args()

    logs_dir = args.output_root / "logs"
    plots_dir = args.output_root / "plots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input_csv)
    df = df_raw[df_raw["seed"].isin(SEEDS)].copy()
    expected_epochs = list(range(5))
    for seed in SEEDS:
        for policy in POLICIES:
            epochs = sorted(df[(df["seed"] == seed) & (df["policy"] == policy)]["epoch"].unique().tolist())
            if epochs != expected_epochs:
                raise RuntimeError(f"incomplete seed={seed}, policy={policy}: epochs={epochs}")

    df.to_csv(logs_dir / "focused_ablation_epoch_logs.csv", index=False)
    (logs_dir / "focused_ablation_3seed_filter.json").write_text(
        json.dumps(
            {
                "source_csv": str(args.input_csv),
                "used_seeds": SEEDS,
                "used_epoch_count": len(expected_epochs),
                "note": "Paper-facing focused-ablation table and plots are filtered to seeds 0,1,2.",
            },
            indent=2,
        )
        + "\n"
    )

    final = df[df["epoch"] == max(expected_epochs)].copy()
    rows = []
    for policy in POLICIES:
        sf = final[final["policy"] == policy]
        sa = df[df["policy"] == policy]
        rows.append(
            {
                "policy": policy,
                "label": LABELS[policy],
                "n_completed_seeds": len(SEEDS),
                "test_acc_mean": sf["test_acc"].mean(),
                "test_acc_std": sf["test_acc"].std(ddof=0),
                "epoch_time_sec_mean": sa["epoch_time_sec"].mean(),
                "epoch_time_sec_std": sa["epoch_time_sec"].std(ddof=0),
                "avg_steps_mean": sf["avg_steps"].mean(),
                "avg_steps_std": sf["avg_steps"].std(ddof=0),
                "cert_bound_mean": sf["cert_bound"].mean(),
                "cert_bound_std": sf["cert_bound"].std(ddof=0),
                "exact_T1_mean": sf["exact_T1"].mean(),
                "exact_T1_std": sf["exact_T1"].std(ddof=0),
                "cap_failure_rate_mean": sf["cap_failure_rate"].mean(),
                "cap_failure_rate_std": sf["cap_failure_rate"].std(ddof=0),
                "rho_star_median_mean": sf["rho_star_median"].mean(),
                "rho_star_median_std": sf["rho_star_median"].std(ddof=0),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(logs_dir / "focused_ablation_summary.csv", index=False)

    b1 = (1.0 + 5e-4) ** 128
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Policy & Acc. & Time/epoch & Avg. steps & Product cert. & Exact $\|T_L\|_1$ & Cap fail. & $\rho_*$ \\",
        r"\midrule",
    ]
    for _, s in summary.iterrows():
        tex.append(
            f"{s['label']} & "
            f"${s['test_acc_mean']:.4f}\\pm{s['test_acc_std']:.4f}$ & "
            f"${s['epoch_time_sec_mean']:.1f}\\pm{s['epoch_time_sec_std']:.1f}$s & "
            f"${s['avg_steps_mean']:.2f}\\pm{s['avg_steps_std']:.2f}$ & "
            f"${s['cert_bound_mean']:.5f}\\pm{s['cert_bound_std']:.5f}$ & "
            f"${s['exact_T1_mean']:.6f}\\pm{s['exact_T1_std']:.8f}$ & "
            f"${100.0*s['cap_failure_rate_mean']:.2f}\\pm{100.0*s['cap_failure_rate_std']:.2f}\\%$ & "
            f"${s['rho_star_median_mean']:.3f}\\pm{s['rho_star_median_std']:.3f}$ \\\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{Focused controller ablation under the CIFAR-10 stress setting. "
        rf"The adaptive and certificate-aware policies use the shared budget "
        rf"$B_1=(1+5\cdot10^{{-4}})^{{128}}={b1:.6f}$. "
        rf"Statistics are computed over seeds $0,1,2$.}}",
        r"\label{tab:focused_controller_ablation}",
        r"\end{table}",
    ]
    (logs_dir / "focused_controller_ablation_table.tex").write_text("\n".join(tex) + "\n")

    plot_specs = [
        ("test_acc", "test accuracy", "focused_test_accuracy.png"),
        ("avg_steps", "average Sinkhorn steps/layer", "focused_avg_steps.png"),
        ("epoch_time_sec", "time per epoch (s)", "focused_time_per_epoch.png"),
        ("cert_bound", r"product certificate $\prod_\ell(1+\varepsilon_c^{(\ell)})$", "focused_product_cert.png"),
        ("exact_T1", r"exact $\|T_L\|_1$", "focused_exact_T1.png"),
        ("rho_star_median", r"median $\rho_*=\sigma_2(P)^2$", "focused_rho_star.png"),
    ]
    plt.rcParams.update({"figure.figsize": (5.7, 3.7), "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    for field, ylabel, fname in plot_specs:
        fig, ax = plt.subplots()
        for policy in POLICIES:
            sub = df[df["policy"] == policy]
            grouped = sub.groupby("epoch")[field].agg(["mean", "std"]).reset_index()
            x = grouped["epoch"].to_numpy(dtype=float) + 1
            y = grouped["mean"].to_numpy(dtype=float)
            sd = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label=LABELS[policy].replace("$", ""))
            ax.fill_between(x, y - sd, y + sd, alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=220, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
