#!/usr/bin/env python3
"""Post-hoc Dobrushin diagnostics for saved stream-mixer matrices.

The script searches for saved arrays/tensors containing per-layer residual
mixers H_res with shape (L, n, n), or with extra leading checkpoint/batch
dimensions ending in (..., L, n, n).  For extra leading dimensions it selects the
last index along every leading axis before the layer axis, documenting that
choice in the output CSV.

No training is run.  If no saved mixer matrices are found, the script still
writes an empty CSV and a LaTeX comment file explaining the expected shape.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "NumPy is required for scripts/dobrushin_diagnostic.py. "
        "Install numpy or run with a Python environment that already provides it."
    ) from exc


DEFAULT_ROOTS = [
    Path("relevance-demo"),
    Path("relevance-demo-dobrushin"),
    Path("paper-extra-experiments"),
    Path("adaptive-budget-ablation"),
    Path("results-stressed-cap80"),
    Path("results-stressed"),
    Path("results-regular"),
    Path("results-cpu-smoke"),
    Path("."),
]

ARRAY_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle"}
NAME_HINTS = re.compile(
    r"(mixer|mixers|h_res|hres|H_res|sinkhorn|diagnostic|diagnostics)",
    re.IGNORECASE,
)

CSV_FIELDS = [
    "run",
    "policy",
    "source_path",
    "object_key",
    "shape_original",
    "choice",
    "L",
    "n",
    "bar_delta",
    "max_row_def",
    "max_col_def_inf",
    "max_col_def_l1",
    "product_certificate",
    "structural_cap",
    "telescoping_certificate",
    "ergodic_certificate",
    "ergodic_uniform_finite_certificate",
    "ergodic_depth_uniform_certificate",
    "exact_gain",
    "ergodic_lt_structural_cap",
    "ergodic_lt_product_certificate",
]


@dataclass
class MixerCandidate:
    source_path: Path
    object_key: str
    array: np.ndarray
    original_shape: tuple[int, ...]
    choice: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute post-hoc Dobrushin/depth-uniform diagnostics from saved mixer tensors."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[str(p) for p in DEFAULT_ROOTS],
        help="Directories to search. Defaults to known result directories plus repo root.",
    )
    parser.add_argument(
        "--csv",
        default="results/dobrushin_diagnostic.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tex",
        default="tables/dobrushin_diagnostic_table.tex",
        help="Output LaTeX table snippet path.",
    )
    parser.add_argument(
        "--comparison-tex",
        default="tables/certificate_comparison_table.tex",
        help="Output policy-level LaTeX certificate-comparison table.",
    )
    parser.add_argument(
        "--hinted-only",
        action="store_true",
        help="Only inspect supported array files whose names contain mixer/sinkhorn hints.",
    )
    return parser.parse_args()


def candidate_files(roots: Iterable[Path], all_arrays: bool = False) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            if path.suffix.lower() not in ARRAY_SUFFIXES:
                continue
            if not all_arrays and not NAME_HINTS.search(path.name):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files)


def load_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix in {".pt", ".pth"}:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PyTorch is required to load .pt/.pth files") from exc
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported suffix: {path.suffix}")


def as_numpy(obj: Any) -> np.ndarray | None:
    """Best-effort conversion for numpy arrays and torch-like tensors."""
    if isinstance(obj, np.ndarray):
        arr = obj
    elif hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        arr = obj.detach().cpu().numpy()
    elif hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        arr = obj.cpu().numpy()
    else:
        return None

    if arr.dtype == object:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.asarray(arr, dtype=float)


def iter_objects(obj: Any, prefix: str = "root", max_depth: int = 8) -> Iterable[tuple[str, Any]]:
    if max_depth < 0:
        return
    yield prefix, obj

    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_objects(value, f"{prefix}.{key}", max_depth - 1)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from iter_objects(value, f"{prefix}[{idx}]", max_depth - 1)


def normalize_mixer_array(arr: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Return a layer stack with shape (L, n, n), if arr looks like mixers."""
    arr = np.asarray(arr, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim < 3:
        return None

    # Preferred convention: (..., L, n, n)
    if arr.shape[-1] == arr.shape[-2] and arr.shape[-1] >= 2:
        if arr.ndim == 3:
            return arr, "used array as (L,n,n)"
        leading = arr.shape[:-2]
        if len(leading) >= 2:
            # Treat the last leading axis as layer index L and all earlier axes
            # as checkpoint/batch/seed dimensions.  Use the final/evaluation
            # slice along those earlier axes.
            index = tuple(-1 for _ in leading[:-1]) + (slice(None), slice(None), slice(None))
            selected = arr[index]
            if selected.ndim == 3 and selected.shape[-1] == selected.shape[-2]:
                return selected, (
                    "selected final index along extra leading dimensions "
                    f"{arr.shape[:-3]}; interpreted remaining shape as (L,n,n)"
                )

    # Fallback convention: two equal square axes somewhere and one layer axis.
    shape = arr.shape
    square_axes: list[tuple[int, int]] = []
    for i in range(arr.ndim):
        for j in range(i + 1, arr.ndim):
            if shape[i] == shape[j] and shape[i] >= 2:
                square_axes.append((i, j))
    for i, j in square_axes:
        other_axes = [k for k in range(arr.ndim) if k not in (i, j)]
        if not other_axes:
            continue
        layer_axis = other_axes[-1]
        perm = other_axes + [i, j]
        moved = np.transpose(arr, perm)
        # Select final indices for any extra axes before the layer axis.
        if len(other_axes) > 1:
            index = tuple(-1 for _ in other_axes[:-1]) + (slice(None), slice(None), slice(None))
            moved = moved[index]
            choice = (
                f"transposed axes {perm}; selected final index along extra axes "
                f"{other_axes[:-1]}; interpreted as (L,n,n)"
            )
        else:
            choice = f"transposed axes {perm}; interpreted as (L,n,n)"
        if moved.ndim == 3 and moved.shape[-1] == moved.shape[-2]:
            return moved, choice
    return None


def extract_candidates(path: Path) -> tuple[list[MixerCandidate], list[str]]:
    messages: list[str] = []
    candidates: list[MixerCandidate] = []
    try:
        obj = load_file(path)
    except Exception as exc:
        return [], [f"{path}: failed to load ({exc})"]

    for key, value in iter_objects(obj):
        arr = as_numpy(value)
        if arr is None:
            continue
        normalized = normalize_mixer_array(arr)
        if normalized is None:
            continue
        layers, choice = normalized
        if layers.ndim != 3 or layers.shape[1] != layers.shape[2]:
            continue
        if not np.all(np.isfinite(layers)):
            messages.append(f"{path}:{key}: skipped non-finite array with shape {arr.shape}")
            continue
        candidates.append(
            MixerCandidate(
                source_path=path,
                object_key=key,
                array=layers,
                original_shape=tuple(arr.shape),
                choice=choice,
            )
        )
    return candidates, messages


def infer_policy(path: Path, key: str) -> str:
    text = f"{path.as_posix()} {key}".lower()
    seed_match = re.search(r"seed(\d+)", text)
    seed_suffix = f" seed {int(seed_match.group(1))}" if seed_match else ""
    for policy in [
        "certificate-aware",
        "adaptive",
        "uniform",
        "fixed",
        "sinkhorn",
        "softmax",
        "row-only",
    ]:
        if policy in text:
            return f"{policy}{seed_suffix}"
    return path.stem


def run_name(path: Path, key: str) -> str:
    parts = list(path.parts)
    if len(parts) >= 3:
        return "/".join(parts[-3:]) + f":{key}"
    return path.as_posix() + f":{key}"


def dobrushin_coefficient(H: np.ndarray) -> float:
    diffs = np.abs(H[:, None, :] - H[None, :, :]).sum(axis=2)
    return 0.5 * float(np.max(diffs))


def metrics(candidate: MixerCandidate) -> dict[str, Any]:
    H_layers = np.asarray(candidate.array, dtype=float)
    L, n, _ = H_layers.shape
    ones = np.ones(n, dtype=float)

    row_defs: list[float] = []
    col_inf_defs: list[float] = []
    col_l1_defs: list[float] = []
    deltas: list[float] = []

    for H in H_layers:
        row = H @ ones - ones
        col = H.T @ ones - ones
        row_defs.append(float(np.max(np.abs(row))))
        col_inf_defs.append(float(np.max(np.abs(col))))
        col_l1_defs.append(float(np.sum(np.abs(col))))
        deltas.append(dobrushin_coefficient(H))

    max_col_def_inf = max(col_inf_defs)
    max_col_def_l1 = max(col_l1_defs)
    bar_delta = max(deltas)
    product_certificate = float(np.prod([1.0 + x for x in col_inf_defs]))
    structural_cap = float(n)

    # Proposition 3.5 uses the norms of the ordered prefix products
    # T_ell = H_ell ... H_1. These cannot be recovered from a single forward
    # vector recursion because matrix multiplication is order-sensitive.
    prefix = np.eye(n, dtype=float)
    telescoping_certificate = 1.0
    for H, col_def in zip(H_layers, col_inf_defs):
        prefix_gain = float(np.max(np.sum(np.abs(prefix), axis=0)))
        telescoping_certificate += prefix_gain * col_def
        prefix = H @ prefix

    # The layerwise finite-horizon certificate retains the actual contraction
    # factors and uses max(z) <= ||z||_1 / 2 for zero-sum defects.
    delta_prefix = 1.0
    ergodic_certificate = 1.0
    for delta, col_l1_def in zip(deltas, col_l1_defs):
        ergodic_certificate += 0.5 * delta_prefix * col_l1_def
        delta_prefix *= delta

    if abs(1.0 - bar_delta) <= 1e-15:
        ergodic_uniform_finite_certificate = 1.0 + 0.5 * max_col_def_l1 * L
        ergodic_depth_uniform_certificate = math.inf
    elif bar_delta < 1.0:
        geometric_sum = (1.0 - bar_delta**L) / (1.0 - bar_delta)
        ergodic_uniform_finite_certificate = float(
            1.0 + 0.5 * max_col_def_l1 * geometric_sum
        )
        ergodic_depth_uniform_certificate = float(
            1.0 + 0.5 * max_col_def_l1 / (1.0 - bar_delta)
        )
    else:
        ergodic_uniform_finite_certificate = math.inf
        ergodic_depth_uniform_certificate = math.inf

    w = ones.copy()
    for H in H_layers[::-1]:
        w = H.T @ w
    exact_gain = float(np.max(w))

    return {
        "run": run_name(candidate.source_path, candidate.object_key),
        "policy": infer_policy(candidate.source_path, candidate.object_key),
        "source_path": candidate.source_path.as_posix(),
        "object_key": candidate.object_key,
        "shape_original": "x".join(map(str, candidate.original_shape)),
        "choice": candidate.choice,
        "L": L,
        "n": n,
        "bar_delta": bar_delta,
        "max_row_def": max(row_defs),
        "max_col_def_inf": max_col_def_inf,
        "max_col_def_l1": max_col_def_l1,
        "product_certificate": product_certificate,
        "structural_cap": structural_cap,
        "telescoping_certificate": telescoping_certificate,
        "ergodic_certificate": ergodic_certificate,
        "ergodic_uniform_finite_certificate": ergodic_uniform_finite_certificate,
        "ergodic_depth_uniform_certificate": ergodic_depth_uniform_certificate,
        "exact_gain": exact_gain,
        "ergodic_lt_structural_cap": bool(ergodic_certificate < structural_cap),
        "ergodic_lt_product_certificate": bool(ergodic_certificate < product_certificate),
    }


def fmt_float(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if math.isinf(val):
        return "inf"
    if val == 0:
        return "0"
    if 0.99 <= abs(val) < 1.01:
        return f"{val:.4f}"
    if 1e-3 <= abs(val) < 1e4:
        return f"{val:.4g}"
    return f"{val:.3e}"


def latex_float(x: Any) -> str:
    text = fmt_float(x)
    if "e" not in text:
        return text
    mant, exp = text.split("e")
    return rf"{mant}\times 10^{{{int(exp)}}}"


def latex_num(x: Any) -> str:
    return rf"\({latex_float(x)}\)"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "% No saved mixer matrices were found. Expected arrays with shape (L,n,n) "
            "or (...,L,n,n).\n"
        )
        return

    lines = [
        r"\begin{center}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Policy & \(L\) & \(n\) & \(\bar\delta\) & \(\max_\ell\|c_\ell\|_1\) & Product cert. & Dobrushin cert. & Exact \(\|T_L\|_1\) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                str(row["policy"]).replace("_", r"\_"),
                int(row["L"]),
                int(row["n"]),
                latex_num(row["bar_delta"]),
                latex_num(row["max_col_def_l1"]),
                latex_num(row["product_certificate"]),
                latex_num(row["ergodic_certificate"]),
                latex_num(row["exact_gain"]),
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\captionof{table}{Post-hoc finite-horizon Dobrushin diagnostics for saved mixer stacks.}",
        r"\label{tab:dobrushin-diagnostic}",
        r"\end{center}",
        "",
    ]
    path.write_text("\n".join(lines))


def policy_name(policy: str) -> str:
    name = re.sub(r"\s+seed\s+\d+$", "", policy).strip()
    if name == "softmax":
        return "row-only softmax"
    if name == "sinkhorn":
        return "Sinkhorn budget"
    return name


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def latex_mean_std(values: list[float]) -> str:
    mean, std = mean_std(values)
    if std <= 1e-12 * max(1.0, abs(mean)):
        return latex_num(mean)
    if abs(mean) >= 1e5:
        exponent = int(math.floor(math.log10(abs(mean))))
        scale = 10.0**exponent
        return rf"\(({mean/scale:.3f}\pm {std/scale:.3f})\times 10^{{{exponent}}}\)"
    if abs(mean) >= 100:
        decimals = 2
    elif std > 0:
        decimals = min(6, max(3, -int(math.floor(math.log10(std))) + 1))
    else:
        decimals = 4
    return rf"\({mean:.{decimals}f}\pm {std:.{decimals}f}\)"


def write_comparison_latex(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("% No saved mixer matrices were found.\n")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(policy_name(str(row["policy"])), []).append(row)

    lines = [
        r"\begin{center}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Mixer policy & Product & Structural & Telescoping & Dobrushin & Exact \\",
        r"\midrule",
    ]
    preferred_order = {"row-only softmax": 0, "Sinkhorn budget": 1}
    for policy in sorted(grouped, key=lambda name: (preferred_order.get(name, 2), name)):
        policy_rows = grouped[policy]
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                policy.replace("_", r"\_"),
                latex_mean_std([float(row["product_certificate"]) for row in policy_rows]),
                latex_mean_std([float(row["structural_cap"]) for row in policy_rows]),
                latex_mean_std([float(row["telescoping_certificate"]) for row in policy_rows]),
                latex_mean_std([float(row["ergodic_certificate"]) for row in policy_rows]),
                latex_mean_std([float(row["exact_gain"]) for row in policy_rows]),
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\captionof{table}{Comparison of backward-gain certificates on the saved final matched-run mixer stacks. Statistics are mean$\pm$std over three seeds.}",
        r"\label{tab:certificate-comparison}",
        r"\end{center}",
        "",
    ]
    path.write_text("\n".join(lines))


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        ("policy", "Policy"),
        ("L", "L"),
        ("n", "n"),
        ("bar_delta", "bar_delta"),
        ("max_col_def_l1", "max ||c||_1"),
        ("product_certificate", "prod cert"),
        ("telescoping_certificate", "tel cert"),
        ("ergodic_certificate", "dob cert"),
        ("exact_gain", "exact"),
        ("ergodic_lt_structural_cap", "dob<n"),
        ("ergodic_lt_product_certificate", "dob<prod"),
    ]
    table = []
    for row in rows:
        rendered = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                rendered.append(fmt_float(value))
            else:
                rendered.append(str(value))
        table.append(rendered)
    widths = [
        max(len(header), *(len(row[i]) for row in table))
        for i, (_, header) in enumerate(columns)
    ]
    header = "  ".join(h.ljust(widths[i]) for i, (_, h) in enumerate(columns))
    print(header)
    print("-" * len(header))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


def main() -> int:
    args = parse_args()
    roots = [Path(x) for x in args.roots]
    searched = [str(root) for root in roots]
    files = candidate_files(roots, all_arrays=not args.hinted_only)

    all_candidates: list[MixerCandidate] = []
    load_messages: list[str] = []
    for path in files:
        candidates, messages = extract_candidates(path)
        all_candidates.extend(candidates)
        load_messages.extend(messages)

    rows = [metrics(candidate) for candidate in all_candidates]
    rows.sort(key=lambda row: (str(row["run"]), str(row["policy"])))

    csv_path = Path(args.csv)
    tex_path = Path(args.tex)
    comparison_tex_path = Path(args.comparison_tex)
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    write_comparison_latex(rows, comparison_tex_path)

    print("Dobrushin/depth-uniform diagnostic")
    print("=" * 40)
    print(f"Searched roots: {', '.join(searched)}")
    print(f"Candidate array files: {len(files)}")
    if files:
        for path in files:
            print(f"  - {path}")
    for msg in load_messages:
        print(f"warning: {msg}", file=sys.stderr)

    if not rows:
        print()
        print("No saved per-layer mixer matrices were found.")
        print("Expected an array/tensor shaped (L,n,n) or (...,L,n,n), where extra leading")
        print("dimensions are checkpoints/batches/seeds and the final slice is used.")
        print(f"Wrote empty CSV header to: {csv_path}")
        print(f"Wrote LaTeX note to: {tex_path}")
        return 0

    print()
    print_table(rows)
    print()
    print(f"CSV written to: {csv_path}")
    print(f"LaTeX table written to: {tex_path}")
    print(f"Certificate comparison written to: {comparison_tex_path}")
    sharper_cap = any(row["ergodic_lt_structural_cap"] for row in rows)
    sharper_prod = any(row["ergodic_lt_product_certificate"] for row in rows)
    print(f"Any Dobrushin certificate sharper than structural cap n? {sharper_cap}")
    print(f"Any Dobrushin certificate sharper than product certificate? {sharper_prod}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
