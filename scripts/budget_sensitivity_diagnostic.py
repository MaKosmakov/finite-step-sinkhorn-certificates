#!/usr/bin/env python3
"""Post-hoc sensitivity of adaptive Sinkhorn stopping to the global budget.

The diagnostic uses the saved final row-only mixer stacks from the matched
transformer run as fixed positive inputs. It changes only the normalization
budget; no model is trained or evaluated for task accuracy.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path

import numpy as np


DEFAULT_BUDGETS = [1.005, 1.012067, 1.03, 1.066075, 1.10, 1.25, 1.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixers",
        default="relevance-demo/mixers",
        help="Directory containing saved final mixer stacks.",
    )
    parser.add_argument(
        "--pattern",
        default="*row_only_softmax*_mixers.npz",
        help="Glob selecting the fixed positive input stacks.",
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--t-max", type=int, default=40)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=500,
        help="Repeated normalization passes used for stable CPU timing.",
    )
    parser.add_argument("--csv", default="results/budget_sensitivity.csv")
    parser.add_argument("--tex", default="tables/budget_sensitivity_table.tex")
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not infer seed from {path.name}")
    return int(match.group(1))


def normalize_stack(
    matrices: np.ndarray, budget: float, t_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    layers = np.asarray(matrices, dtype=np.float64)
    L = layers.shape[0]
    remaining_log_budget = math.log(budget)
    outputs: list[np.ndarray] = []
    steps: list[int] = []
    defects: list[float] = []
    failures = 0

    for ell, matrix in enumerate(layers):
        remaining_layers = L - ell
        target = math.expm1(max(remaining_log_budget, 0.0) / remaining_layers)
        H = matrix.copy()
        defect = math.inf
        for step in range(1, t_max + 1):
            H /= H.sum(axis=0, keepdims=True)
            H /= H.sum(axis=1, keepdims=True)
            defect = float(np.max(np.abs(H.sum(axis=0) - 1.0)))
            if defect <= target:
                break
        if defect > target:
            failures += 1
        remaining_log_budget -= math.log1p(defect)
        outputs.append(H.copy())
        steps.append(step)
        defects.append(defect)

    return (
        np.stack(outputs),
        np.asarray(steps, dtype=int),
        np.asarray(defects, dtype=float),
        failures / L,
    )


def exact_gain(layers: np.ndarray) -> float:
    w = np.ones(layers.shape[-1], dtype=np.float64)
    for H in layers[::-1]:
        w = H.T @ w
    return float(np.max(w))


def benchmark_ms(
    matrices: np.ndarray, budget: float, t_max: int, repeats: int
) -> float:
    for _ in range(5):
        normalize_stack(matrices, budget, t_max)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        normalize_stack(matrices, budget, t_max)
        samples.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(samples))


def evaluate(path: Path, budget: float, t_max: int, repeats: int) -> dict[str, float | int | str]:
    with np.load(path) as saved:
        layers = np.asarray(saved["H_res"], dtype=np.float64)
    if layers.ndim != 3 or layers.shape[1] != layers.shape[2]:
        raise ValueError(f"Expected (L,n,n) in {path}, found {layers.shape}")

    normalized, steps, defects, failure_rate = normalize_stack(layers, budget, t_max)
    return {
        "seed": seed_from_path(path),
        "source": path.as_posix(),
        "budget": budget,
        "L": layers.shape[0],
        "n": layers.shape[1],
        "initial_tolerance": math.expm1(math.log(budget) / layers.shape[0]),
        "avg_steps": float(steps.mean()),
        "time_ms_per_stack": benchmark_ms(layers, budget, t_max, repeats),
        "product_certificate": float(np.prod(1.0 + defects)),
        "exact_gain": exact_gain(normalized),
        "max_col_def_inf": float(defects.max()),
        "cap_failure_rate": failure_rate,
    }


def mean_std(rows: list[dict[str, float | int | str]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def latex_value(mean: float, std: float, scientific: bool = False) -> str:
    if scientific:
        return rf"\({mean:.2e}\pm {std:.1e}\)"
    return rf"\({mean:.4f}\pm {std:.4f}\)"


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[float, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["budget"]), []).append(row)

    lines = [
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Budget \(B_1\) & Avg. steps & CPU ms/stack & Product cert. & Exact \(\|T_L\|_1\) & Cap fail. \\",
        r"\midrule",
    ]
    for budget in sorted(grouped):
        group = grouped[budget]
        steps = mean_std(group, "avg_steps")
        timing = mean_std(group, "time_ms_per_stack")
        product = mean_std(group, "product_certificate")
        exact = mean_std(group, "exact_gain")
        failure = mean_std(group, "cap_failure_rate")
        failure_text = (
            r"\(0\%\)"
            if abs(failure[0]) < 1e-15 and abs(failure[1]) < 1e-15
            else rf"\({100*failure[0]:.2f}\pm {100*failure[1]:.2f}\%\)"
        )
        lines.append(
            rf"{budget:.6g} & {latex_value(*steps)} & {latex_value(*timing)} & "
            rf"{latex_value(*product)} & {latex_value(*exact)} & "
            rf"{failure_text} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\captionof{table}{Post-hoc global-budget sensitivity on saved final matched-run mixer stacks. Timing is for normalization of one 24-layer stack on a single CPU process and excludes file loading. Statistics are mean$\pm$std over three seeds.}",
        r"\label{tab:budget-sensitivity}",
        r"\end{center}",
        "",
    ]
    path.write_text("\n".join(lines))


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    grouped: dict[float, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["budget"]), []).append(row)
    print("B1       avg_steps  ms/stack  product   exact     cap_fail")
    for budget in sorted(grouped):
        group = grouped[budget]
        print(
            f"{budget:<8.6g} "
            f"{mean_std(group, 'avg_steps')[0]:>9.4f}  "
            f"{mean_std(group, 'time_ms_per_stack')[0]:>8.4f}  "
            f"{mean_std(group, 'product_certificate')[0]:>8.5f}  "
            f"{mean_std(group, 'exact_gain')[0]:>8.6f}  "
            f"{100*mean_std(group, 'cap_failure_rate')[0]:>7.2f}%"
        )


def main() -> int:
    args = parse_args()
    paths = sorted(Path(args.mixers).glob(args.pattern))
    if not paths:
        raise SystemExit(
            f"No mixer stacks matched {Path(args.mixers) / args.pattern}. "
            "Expected .npz files containing H_res with shape (L,n,n)."
        )
    if any(budget <= 1.0 for budget in args.budgets):
        raise SystemExit("Every budget must be greater than one.")

    rows = [
        evaluate(path, budget, args.t_max, args.timing_repeats)
        for budget in args.budgets
        for path in paths
    ]
    write_csv(rows, Path(args.csv))
    write_tex(rows, Path(args.tex))
    print_summary(rows)
    print(f"CSV written to: {args.csv}")
    print(f"LaTeX table written to: {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
