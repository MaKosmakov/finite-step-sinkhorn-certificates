#!/usr/bin/env python3
"""Regenerate paper-facing CIFAR/Sinkhorn plots from epoch_logs.csv."""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Callable, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib-cache"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


COLORS = {"sinkhorn": "#1f77b4", "softmax": "#d95f02"}
MARKERS = {"sinkhorn": "o", "softmax": "s"}
LABELS = {"sinkhorn": "Sinkhorn early stop", "softmax": "row-softmax"}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (4.8, 3.2),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.7,
            "lines.markersize": 4,
        }
    )


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def epoch_stats(data: pd.DataFrame, field: str) -> np.ndarray | None:
    rows = []
    for epoch, group in data.groupby("epoch"):
        vals = finite(group[field].dropna())
        if vals.size:
            rows.append(
                (
                    epoch + 1,
                    vals.mean(),
                    vals.std(ddof=0),
                    np.percentile(vals, 25),
                    np.percentile(vals, 75),
                )
            )
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def savefig(out_dir: str, name: str) -> None:
    plt.tight_layout(pad=0.8)
    plt.savefig(os.path.join(out_dir, name), bbox_inches="tight")
    plt.close()


def plot_methods(
    df: pd.DataFrame,
    out_dir: str,
    field: str,
    ylabel: str,
    title: str,
    name: str,
    *,
    methods: tuple[str, ...] = ("sinkhorn", "softmax"),
    transform: Callable[[pd.Series], np.ndarray] | None = None,
    logy: bool = False,
    hline: tuple[float, str, str] | None = None,
    note: str | None = None,
) -> None:
    if field not in df.columns:
        return

    fig, ax = plt.subplots()
    plotted = False
    for method in methods:
        sub = df[df["model_type"] == method].copy()
        if sub.empty:
            continue
        if transform is not None:
            sub[field] = transform(sub[field].astype(float))
        stats = epoch_stats(sub, field)
        if stats is None:
            continue
        x, mean, _, q25, q75 = stats.T
        color = COLORS.get(method)

        for _, seed_group in sub.groupby("seed"):
            vals = seed_group.sort_values("epoch")[["epoch", field]].dropna()
            if vals.empty:
                continue
            sx = vals["epoch"].to_numpy(dtype=float) + 1
            sy = vals[field].to_numpy(dtype=float)
            ax.plot(sx, sy, color=color, alpha=0.22, linewidth=1.0)
            ax.scatter(sx, sy, color=color, alpha=0.30, s=14)

        ax.plot(
            x,
            mean,
            marker=MARKERS.get(method, "o"),
            color=color,
            label=LABELS.get(method, method),
        )
        if len(x) > 1 and np.any(q75 > q25):
            ax.fill_between(x, q25, q75, color=color, alpha=0.12, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    if logy:
        ax.set_yscale("log")
    if hline is not None:
        y, label, linestyle = hline
        ax.axhline(y, color="0.35", linestyle=linestyle, linewidth=1.1, label=label)
    if note is not None:
        ax.text(
            0.98,
            0.05,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="0.25",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=True, loc="best")
    savefig(out_dir, name)


def cap_failure_note(df: pd.DataFrame, cap: int) -> str:
    sink = df[df["model_type"] == "sinkhorn"]
    if "sink_hit_rate" not in sink.columns or sink.empty:
        return f"cap {cap}"
    hit = sink["sink_hit_rate"].dropna()
    if hit.empty:
        return f"cap {cap}"
    fail = max(0.0, 1.0 - float(hit.iloc[-1]))
    return f"cap {cap}; final cap failure {fail:.3g}"


def make_plots(csv_path: str, out_dir: str, tolerance: float, cap: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    configure_matplotlib()

    plot_methods(
        df,
        out_dir,
        "kappa_median",
        r"median observed dynamic range $\kappa_{\mathrm{obs}}$",
        "",
        "kappa_obs_log.png",
        logy=True,
    )
    plot_methods(
        df,
        out_dir,
        "kappa_median",
        r"median observed dynamic range $\kappa_{\mathrm{obs}}$",
        "",
        "kappa_obs.png",
    )
    plot_methods(
        df,
        out_dir,
        "tstop_median",
        r"median $t_{\mathrm{stop}}$",
        "",
        "tstop_hist_early.png",
        methods=("sinkhorn",),
        note=cap_failure_note(df, cap),
    )
    plot_methods(
        df,
        out_dir,
        "max_eps_c",
        r"$\max_\ell \varepsilon_c^{(\ell)}$",
        "",
        "max_eps_c.png",
        methods=("sinkhorn",),
        logy=True,
        hline=(tolerance, rf"tolerance {tolerance:g}", "--"),
    )
    plot_methods(
        df,
        out_dir,
        "cert_bound",
        r"$\log_{10}$ product certificate",
        "",
        "cert_bound.png",
        transform=lambda series: np.log10(np.asarray(series, dtype=float)),
    )
    plot_methods(
        df,
        out_dir,
        "exact_T1",
        r"exact $\|T_L\|_1$",
        "",
        "exact_T1.png",
        hline=(1.0, "no amplification", "--"),
    )
    plot_methods(
        df,
        out_dir,
        "rho_star_median",
        r"median $\rho_*=\sigma_2(P)^2$",
        "",
        "rho_star_median.png",
        methods=("sinkhorn",),
    )
    plot_methods(
        df,
        out_dir,
        "empirical_step_ratio_median",
        "median final-step error ratio",
        "",
        "empirical_step_ratio.png",
        methods=("sinkhorn",),
    )
    plot_methods(
        df,
        out_dir,
        "test_acc",
        "test accuracy",
        "",
        "test_accuracy.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results-stressed-cap80/logs/epoch_logs.csv",
        help="Path to the runner's epoch_logs.csv.",
    )
    parser.add_argument(
        "--out_dir",
        default="results-stressed-cap80/plots",
        help="Directory where regenerated PNG plots are written.",
    )
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--cap", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plots(args.csv, args.out_dir, args.tolerance, args.cap)
    print(f"Wrote cleaned paper plots to {args.out_dir}")


if __name__ == "__main__":
    main()
