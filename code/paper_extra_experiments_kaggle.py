# Extra paper experiments for finite-step Sinkhorn.
#
# Runs:
#   A. focused policy ablation:
#      fixed cap vs uniform early stop vs adaptive log budget vs certificate-aware penalty
#   B. temperature-clipping design-rule grid:
#      measured column error vs the conservative Psi_n certificate
#   C. local-predictor curves for trained matrices:
#      measured curve vs kappa-only bound, matrix-adaptive Hilbert bound,
#      first-order local predictor, and second-order predictor.
#
# This file is Kaggle-ready. It expects adaptive_budget_ablation_kaggle.py
# in the same directory, because it reuses the CIFAR model and Sinkhorn code.

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adaptive_budget_ablation_kaggle as base


# =========================
# Kaggle-editable settings
# =========================
CONFIG = {
    # Use [0] for a quick diagnostic. The paper-facing three-seed focused
    # ablation is run by paper_extra_experiments_3seeds_5epochs_kaggle.py.
    "seeds": [0, 1, 2],
    "epochs": 5,

    # CIFAR / runtime.
    "batch_size": 256,
    "num_workers": 4,
    "amp": True,
    "data_parallel": False,
    "data_root": "/kaggle/input/cifar10-python",
    "out_dir": "/kaggle/working/paper-extra-experiments",

    # Debug knobs. Keep None for the real run.
    "max_train_batches": None,
    "max_eval_batches": None,

    # Stress profile used in the paper.
    "depth": 128,
    "n_streams": 8,
    "d": 128,
    "beta_sinkhorn": 3.0,
    "init_scale": 2.5,
    "sinkhorn_t_max": 80,
    "sinkhorn_eps_tol": 5e-4,
    "sinkhorn_clip_logits": 10.0,
    "lr": 0.02,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    # Certificate-aware row. It uses the adaptive controller plus the objective
    # in the paper: task loss + lambda_cert R_budget + lambda_kappa kappa penalty.
    "cert_penalty_base_policy": "adaptive",
    "lambda_cert": 1.0,
    "lambda_kappa": 1e-4,
    "kappa_max": 20.0,

    # Temperature-clipping grid.
    "run_temperature_grid": True,
    "temperature_beta_R_values": [0.5, 1.0, 2.0, 4.0],
    "temperature_t_values": [5, 10, 20, 40, 80],
    "temperature_trials": 64,
    "temperature_matrix_n": 8,
    "temperature_seed": 123,

    # Local predictor plots, extracted from the trained adaptive model.
    "run_local_predictor": True,
    "local_predictor_policy": "adaptive",
    "local_predictor_layers": [0, 64, 127],
    "local_predictor_t_max": 80,
    "local_predictor_limit_iters": 700,
}

POLICIES = ["fixed", "uniform", "adaptive", "cert_penalty"]
POLICY_LABELS = {
    "fixed": r"Fixed $t_{\max}=80$",
    "uniform": "Uniform tolerance",
    "adaptive": "Adaptive log-budget",
    "cert_penalty": "Certificate-aware penalty",
}


def ensure_dirs(out_dir: str) -> Tuple[str, str]:
    logs_dir = os.path.join(out_dir, "logs")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return logs_dir, plots_dir


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def project(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean()


def psi_n(delta: float, n: int) -> float:
    delta = max(float(delta), 0.0)
    if delta > 60.0:
        return float(n - 1)
    r = math.exp(delta)
    return float((n - 1) * (r - 1.0) / ((n - 1) + r))


def gamma_n(eta: float, n: int) -> float:
    eta = max(float(eta), 0.0)
    a = math.sqrt((n - 1) / n) * eta
    if a > 60.0:
        return float(n - 1)
    ea = math.exp(a)
    eb = math.exp(-a / max(n - 1, 1))
    return float(n * ea / (ea + (n - 1) * eb) - 1.0)


def tex_num(x: float, *, floor_label: bool = False) -> str:
    if not math.isfinite(float(x)):
        return "---"
    x = float(x)
    if x == 0.0 and floor_label:
        return r"$<10^{-15}$"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"${x:.2e}$"
    return f"${x:.3g}$"


def rho_star_from_matrix(A: torch.Tensor) -> float:
    vals = torch.linalg.svdvals(A.detach().to(dtype=torch.float64, device="cpu"))
    vals, _ = torch.sort(vals, descending=True)
    if vals.numel() < 2:
        return 0.0
    return float((vals[1] ** 2).item())


def positive_log_range(mixer: base.SinkhornMixer) -> torch.Tensor:
    z = mixer.beta * mixer.W
    if mixer.clip_logits > 0:
        z = z.clamp(-mixer.clip_logits, mixer.clip_logits)
    return z.max() - z.min()


def differentiable_certificate_penalty(model: nn.Module, cfg: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable version of the paper penalty for the cert_penalty row."""
    m = base.unwrap(model)
    device = next(m.parameters()).device
    dtype = next(m.parameters()).dtype
    log_budget_target = torch.tensor(float(m.log_budget), device=device, dtype=dtype)
    log_kappa_max = torch.tensor(math.log(float(cfg["kappa_max"])), device=device, dtype=dtype)

    used_log_budget = 0.0
    log_terms: List[torch.Tensor] = []
    kappa_terms: List[torch.Tensor] = []
    step_values: List[float] = []

    for ell, layer in enumerate(m.layers):
        eps_target = None
        if m.policy == "adaptive":
            remaining_layers = m.depth - ell
            remaining_budget = max(m.log_budget - used_log_budget, 0.0)
            eps_target = math.expm1(remaining_budget / max(remaining_layers, 1))

        A = layer.mixer(eps_target=eps_target)
        col_err = torch.amax(torch.abs(A.sum(dim=0) - 1.0))
        log_terms.append(torch.log1p(col_err))
        step_values.append(float(layer.mixer.last_t_stop))

        if m.policy == "adaptive":
            used_log_budget += math.log1p(max(float(layer.mixer.last_eps_c), 0.0))

        log_kappa = positive_log_range(layer.mixer)
        kappa_terms.append(F.relu(log_kappa - log_kappa_max) ** 2)

    c_prod = torch.stack(log_terms).sum()
    r_budget = F.relu(c_prod - log_budget_target) ** 2
    r_kappa = torch.stack(kappa_terms).mean()
    penalty = float(cfg["lambda_cert"]) * r_budget + float(cfg["lambda_kappa"]) * r_kappa
    return penalty, {
        "cert_penalty_budget": float(r_budget.detach().cpu().item()),
        "cert_penalty_kappa": float(r_kappa.detach().cpu().item()),
        "cert_penalty_log_prod": float(c_prod.detach().cpu().item()),
        "cert_penalty_avg_steps_extra": float(np.mean(step_values)) if step_values else float("nan"),
    }


def make_policy_models(cfg: Dict[str, object], device: torch.device) -> Dict[str, nn.Module]:
    base_model = base.MixerNet(cfg, policy="fixed").to(device)
    base_state = copy.deepcopy(base_model.state_dict())
    models: Dict[str, nn.Module] = {"fixed": base_model}

    for policy in ["uniform", "adaptive"]:
        model = base.MixerNet(cfg, policy=policy).to(device)
        model.load_state_dict(base_state)
        models[policy] = model

    cert_base_policy = str(cfg.get("cert_penalty_base_policy", "adaptive"))
    cert_model = base.MixerNet(cfg, policy=cert_base_policy).to(device)
    cert_model.load_state_dict(base_state)
    models["cert_penalty"] = cert_model
    return models


def train_one_epoch(
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    scalers: Dict[str, torch.cuda.amp.GradScaler],
    loader,
    device: torch.device,
    cfg: Dict[str, object],
) -> Dict[str, Tuple[float, float, float, Dict[str, float]]]:
    for model in models.values():
        model.train(True)

    totals = {p: {"loss": 0.0, "acc": 0.0, "n": 0, "time": 0.0, "aux": []} for p in POLICIES}
    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    max_batches = cfg.get("max_train_batches")

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = int(x.shape[0])

        for policy in POLICIES:
            t0 = time.perf_counter()
            model = models[policy]
            opt = optimizers[policy]
            scaler = scalers[policy]
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                task_loss = F.cross_entropy(logits, y)
                loss = task_loss
                aux = {}
                if policy == "cert_penalty":
                    penalty, aux = differentiable_certificate_penalty(model, cfg)
                    loss = task_loss + penalty

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            totals[policy]["time"] += time.perf_counter() - t0
            totals[policy]["loss"] += float(loss.detach().item()) * bs
            totals[policy]["acc"] += base.acc_value(logits.detach(), y) * bs
            totals[policy]["n"] += bs
            if aux:
                totals[policy]["aux"].append(aux)

    out: Dict[str, Tuple[float, float, float, Dict[str, float]]] = {}
    for policy, v in totals.items():
        aux_summary: Dict[str, float] = {}
        if v["aux"]:
            keys = v["aux"][0].keys()
            aux_summary = {k: float(np.mean([a[k] for a in v["aux"]])) for k in keys}
        out[policy] = (v["loss"] / v["n"], v["acc"] / v["n"], v["time"], aux_summary)
    return out


@torch.no_grad()
def collect_extended_stats(model: nn.Module) -> Dict[str, float]:
    stats = dict(base.collect_policy_stats(model))
    m = base.unwrap(model)
    rho_values: List[float] = []
    for layer in m.layers:
        rho_values.append(rho_star_from_matrix(layer.mixer.fixed_matrix()))
    stats["rho_star_median"] = float(np.nanmedian(np.array(rho_values, dtype=float))) if rho_values else float("nan")
    stats["rho_star_mean"] = float(np.nanmean(np.array(rho_values, dtype=float))) if rho_values else float("nan")
    return stats


def completed_seed_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    final_epoch = int(df["epoch"].max())
    complete_seeds: List[int] = []
    for seed, seed_df in df.groupby("seed"):
        seed_final = seed_df[seed_df["epoch"] == final_epoch]
        if set(seed_final["policy"]) >= set(POLICIES):
            complete_seeds.append(int(seed))
    if complete_seeds:
        return df[df["seed"].isin(complete_seeds)].copy(), complete_seeds
    return df.copy(), []


def write_focused_ablation_outputs(rows: List[Dict[str, object]], cfg: Dict[str, object]) -> None:
    out_dir = str(cfg["out_dir"])
    logs_dir, plots_dir = ensure_dirs(out_dir)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(logs_dir, "focused_ablation_epoch_logs.csv"), index=False)

    df_summary, complete_seeds = completed_seed_frame(df)
    final_epoch = int(df_summary["epoch"].max())
    final = df_summary[df_summary["epoch"] == final_epoch].copy()

    summary_rows: List[Dict[str, object]] = []
    for policy in POLICIES:
        sf = final[final["policy"] == policy]
        sa = df_summary[df_summary["policy"] == policy]
        summary_rows.append({
            "policy": policy,
            "label": POLICY_LABELS[policy],
            "n_completed_seeds": len(complete_seeds),
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
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(logs_dir, "focused_ablation_summary.csv"), index=False)

    tex: List[str] = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\scriptsize")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(r"\begin{tabular}{lccccccc}")
    tex.append(r"\toprule")
    tex.append(r"Policy & Acc. & Time/epoch & Avg. steps & Product cert. & Exact $\|T_L\|_1$ & Cap fail. & $\rho_*$ \\")
    tex.append(r"\midrule")
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
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")
    b1 = (1.0 + float(cfg["sinkhorn_eps_tol"])) ** int(cfg["depth"])
    tex.append(
        rf"\caption{{Focused controller ablation under the CIFAR-10 stress setting. "
        rf"The adaptive and certificate-aware policies use the shared budget "
        rf"$B_1=(1+5\cdot10^{{-4}})^{{128}}={b1:.6f}$. "
        rf"Statistics are computed over completed seeds {complete_seeds if complete_seeds else 'available in the log'}.}}"
    )
    tex.append(r"\label{tab:focused_controller_ablation}")
    tex.append(r"\end{table}")
    with open(os.path.join(logs_dir, "focused_controller_ablation_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    plot_specs = [
        ("test_acc", "test accuracy", "focused_test_accuracy.png"),
        ("avg_steps", "average Sinkhorn steps/layer", "focused_avg_steps.png"),
        ("epoch_time_sec", "time per epoch (s)", "focused_time_per_epoch.png"),
        ("cert_bound", r"product certificate $\prod_\ell(1+\varepsilon_c^{(\ell)})$", "focused_product_cert.png"),
        ("exact_T1", r"exact $\|T_L\|_1$", "focused_exact_T1.png"),
        ("rho_star_median", r"median $\rho_*=\sigma_2(P)^2$", "focused_rho_star.png"),
    ]
    plt.rcParams.update({"figure.figsize": (5.7, 3.7), "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    plot_df = df_summary if len(df_summary) else df
    for field, ylabel, fname in plot_specs:
        fig, ax = plt.subplots()
        for policy in POLICIES:
            sub = plot_df[plot_df["policy"] == policy]
            grouped = sub.groupby("epoch")[field].agg(["mean", "std"]).reset_index()
            x = grouped["epoch"].to_numpy(dtype=float) + 1
            y = grouped["mean"].to_numpy(dtype=float)
            sd = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label=POLICY_LABELS[policy].replace("$", ""))
            if len(x) > 1:
                ax.fill_between(x, y - sd, y + sd, alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=220, bbox_inches="tight")
        plt.close(fig)


def run_focused_ablation(cfg: Dict[str, object], device: torch.device) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, nn.Module]]]:
    all_rows: List[Dict[str, object]] = []
    final_models_by_seed: Dict[int, Dict[str, nn.Module]] = {}

    for seed in list(cfg["seeds"]):
        seed = int(seed)
        print(f"\n=== focused ablation seed {seed} ===")
        base.set_global_seed(seed)
        train_loader, test_loader, sampler = base.get_loaders(cfg, seed)
        models = make_policy_models(cfg, device)

        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
        if bool(cfg.get("data_parallel", False)) and n_gpus >= 2:
            models = {p: nn.DataParallel(m) for p, m in models.items()}

        optimizers = {
            p: torch.optim.SGD(
                m.parameters(),
                lr=float(cfg["lr"]),
                momentum=float(cfg["momentum"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            for p, m in models.items()
        }
        scalers = {
            p: torch.cuda.amp.GradScaler(enabled=(bool(cfg["amp"]) and device.type == "cuda"))
            for p in POLICIES
        }

        for epoch in range(int(cfg["epochs"])):
            sampler.set_epoch(epoch)
            train_stats = train_one_epoch(models, optimizers, scalers, train_loader, device, cfg)
            for policy in POLICIES:
                test_loss, test_acc, eval_time = base.evaluate(models[policy], test_loader, device, cfg)
                stats = collect_extended_stats(models[policy])
                train_loss, train_acc, train_time, aux = train_stats[policy]
                row = {
                    "seed": seed,
                    "epoch": int(epoch),
                    "policy": policy,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "train_time_sec": float(train_time),
                    "eval_time_sec": float(eval_time),
                    "epoch_time_sec": float(train_time + eval_time),
                    **{k: float(v) for k, v in stats.items()},
                    **{k: float(v) for k, v in aux.items()},
                }
                all_rows.append(row)
                print(
                    f"epoch {epoch:02d} {policy:13s} "
                    f"acc={test_acc:.3f} steps={stats['avg_steps']:.2f} "
                    f"cert={stats['cert_bound']:.4g} exactT1={stats['exact_T1']:.4g} "
                    f"rho_star={stats['rho_star_median']:.3g} time={train_time + eval_time:.1f}s"
                )
            write_focused_ablation_outputs(all_rows, cfg)

        final_models_by_seed[seed] = models

    write_focused_ablation_outputs(all_rows, cfg)
    return all_rows, final_models_by_seed


# =============================================================================
# Temperature-clipping grid
# =============================================================================

@torch.no_grad()
def full_sinkhorn_matrix(M: torch.Tensor, t_max: int) -> torch.Tensor:
    X = M.clone()
    for _ in range(int(t_max)):
        X = X / X.sum(dim=0, keepdim=True).clamp_min(1e-20)
        X = X / X.sum(dim=1, keepdim=True).clamp_min(1e-20)
    return X


@torch.no_grad()
def column_error(A: torch.Tensor) -> float:
    ones = torch.ones(A.shape[0], dtype=A.dtype, device=A.device)
    return float((A.sum(dim=0) - ones).abs().max().item())


def run_temperature_grid(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    print("\n=== temperature-clipping grid ===")
    out_dir = str(cfg["out_dir"])
    logs_dir, plots_dir = ensure_dirs(out_dir)
    rng = torch.Generator(device="cpu").manual_seed(int(cfg["temperature_seed"]))
    n = int(cfg["temperature_matrix_n"])
    trials = int(cfg["temperature_trials"])
    rows: List[Dict[str, object]] = []

    for beta_R in list(cfg["temperature_beta_R_values"]):
        beta_R = float(beta_R)
        for t_max in list(cfg["temperature_t_values"]):
            t_max = int(t_max)
            eps_values: List[float] = []
            for _ in range(trials):
                C = 2.0 * torch.rand((n, n), generator=rng, dtype=torch.float64) - 1.0
                M = torch.exp(beta_R * C)
                H = full_sinkhorn_matrix(M, t_max)
                eps_values.append(column_error(H))

            delta_cert = 2.0 * beta_R * (math.tanh(beta_R) ** (2 * t_max - 1))
            cert = psi_n(delta_cert, n)
            rows.append({
                "beta_R": beta_R,
                "t_max": t_max,
                "n": n,
                "trials": trials,
                "measured_eps_mean": float(np.mean(eps_values)),
                "measured_eps_median": float(np.median(eps_values)),
                "measured_eps_max": float(np.max(eps_values)),
                "certificate": float(cert),
                "max_over_cert": float(np.max(eps_values) / cert) if cert > 0 else float("nan"),
                "certificate_delta": float(delta_cert),
            })
            print(f"betaR={beta_R:g} t={t_max:02d} max_eps={np.max(eps_values):.3g} cert={cert:.3g}")

    write_csv(os.path.join(logs_dir, "temperature_clipping_grid.csv"), rows)
    df = pd.DataFrame(rows)

    tex: List[str] = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\scriptsize")
    tex.append(r"\begin{tabular}{ccccc}")
    tex.append(r"\toprule")
    tex.append(r"$\beta R$ & $t_{\max}$ & max measured $\varepsilon_c$ & certificate & max/cert. \\")
    tex.append(r"\midrule")
    for _, r in df.iterrows():
        cert_value = float(r["certificate"])
        ratio_value = float(r["max_over_cert"]) if math.isfinite(float(r["max_over_cert"])) else float("nan")
        tex.append(
            f"{r['beta_R']:.1f} & {int(r['t_max'])} & "
            f"{tex_num(r['measured_eps_max'])} & "
            f"{tex_num(cert_value, floor_label=True)} & "
            f"{tex_num(ratio_value)} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\caption{Temperature--clipping grid for the design rule "
               r"$\Psi_n(2\beta R\tanh(\beta R)^{2t_{\max}-1})$. "
               r"Measured errors are maxima over random clipped logits with $n=8$. "
               r"Entries shown as $<10^{-15}$ are below the numerical floor of this double-precision diagnostic.}")
    tex.append(r"\label{tab:temperature_clipping_grid}")
    tex.append(r"\end{table}")
    with open(os.path.join(logs_dir, "temperature_clipping_grid_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for beta_R, sub in df.groupby("beta_R"):
        sub = sub.sort_values("t_max")
        ax.plot(sub["t_max"], sub["measured_eps_max"].clip(lower=1e-16), marker="o", label=rf"measured max $\beta R={beta_R:g}$")
        ax.plot(sub["t_max"], sub["certificate"].clip(lower=1e-16), linestyle="--", marker="x", label=rf"cert. $\beta R={beta_R:g}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_{\max}$")
    ax.set_ylabel(r"column error / certificate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "temperature_clipping_grid.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return rows


# =============================================================================
# Local predictor curves
# =============================================================================

@torch.no_grad()
def projective_diameter(A: torch.Tensor) -> float:
    L = torch.log(A.to(dtype=torch.float64, device="cpu"))
    vals = L[:, None, :, None] + L[None, :, None, :] - L[:, None, None, :] - L[None, :, :, None]
    return float(vals.max().item())


@torch.no_grad()
def sinkhorn_curve(A: torch.Tensor, t_max: int) -> List[float]:
    X = A.to(dtype=torch.float64, device="cpu").clone()
    eps: List[float] = []
    for _ in range(int(t_max)):
        X = X / X.sum(dim=0, keepdim=True).clamp_min(1e-300)
        X = X / X.sum(dim=1, keepdim=True).clamp_min(1e-300)
        eps.append(column_error(X))
    return eps


@torch.no_grad()
def sinkhorn_limit(A: torch.Tensor, iters: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = A.to(dtype=torch.float64, device="cpu")
    n = A.shape[0]
    r = torch.ones(n, dtype=torch.float64)
    c = torch.ones(n, dtype=torch.float64)
    for _ in range(int(iters)):
        c = 1.0 / (A.T @ r).clamp_min(1e-300)
        r = 1.0 / (A @ c).clamp_min(1e-300)
    P = torch.diag(r) @ A @ torch.diag(c)
    return P, r, c


def centered_exp_error(ell: torch.Tensor) -> float:
    v = torch.softmax(ell, dim=0) * ell.numel()
    return float((v - 1.0).abs().max().item())


@torch.no_grad()
def local_predictor_rows(A: torch.Tensor, t_max: int, limit_iters: int) -> Tuple[List[Dict[str, object]], float]:
    A = A.to(dtype=torch.float64, device="cpu")
    n = A.shape[0]
    measured = sinkhorn_curve(A, t_max)

    P, r_star, _c_star = sinkhorn_limit(A, limit_iters)
    K = P @ P.T
    I = torch.eye(n, dtype=torch.float64)
    x0 = project(-torch.log(r_star))
    rho_star = rho_star_from_matrix(P)

    kappa = float(A.max().item() / A.min().item())
    q = (kappa - 1.0) / (kappa + 1.0) if kappa > 1.0 else 0.0
    delta = projective_diameter(A)
    tau = math.tanh(delta / 4.0)
    P1 = A / A.sum(dim=0, keepdim=True).clamp_min(1e-300)
    row_sums = P1.sum(dim=1)
    delta0 = float(torch.log(row_sums.max() / row_sums.min()).item())

    def QF(x: torch.Tensor) -> torch.Tensor:
        ptx = P.T @ x
        kx = K @ x
        return -P @ (ptx ** 2) + 0.5 * (K @ (x ** 2)) + 0.5 * (kx ** 2)

    def QL(x: torch.Tensor) -> torch.Tensor:
        ptx = P.T @ x
        return 0.5 * (P.T @ (x ** 2)) - 0.5 * (ptx ** 2)

    x1: List[torch.Tensor] = [x0]
    x2: List[torch.Tensor] = [x0]
    for t in range(1, int(t_max) + 1):
        prev_x1 = x1[-1]
        x2.append(K @ x2[-1] + project(QF(prev_x1)))
        x1.append(K @ prev_x1)

    rows: List[Dict[str, object]] = []
    for t in range(1, int(t_max) + 1):
        ell1 = project(P.T @ ((K - I) @ x1[t - 1]))
        ell2 = project(P.T @ x2[t] + QL(x1[t]) - P.T @ x2[t - 1] - QL(x1[t - 1]))
        kappa_bound = psi_n(math.log(kappa) * (q ** (2 * t - 1)), n) if t >= 1 else float("nan")
        adaptive_bound = psi_n(delta0 * (tau ** (2 * t - 1)), n)
        rows.append({
            "t": t,
            "measured_eps_c": measured[t - 1],
            "kappa_bound": kappa_bound,
            "matrix_adaptive_bound": adaptive_bound,
            "local_first_order": centered_exp_error(ell1),
            "local_second_order": centered_exp_error(ell2),
            "rho_star": rho_star,
            "kappa": kappa,
            "projective_delta": delta,
            "tau": tau,
            "delta0": delta0,
        })
    return rows, rho_star


def run_local_predictor_export(models_by_seed: Dict[int, Dict[str, nn.Module]], cfg: Dict[str, object]) -> List[Dict[str, object]]:
    print("\n=== local predictor curves ===")
    out_dir = str(cfg["out_dir"])
    logs_dir, plots_dir = ensure_dirs(out_dir)
    rows_all: List[Dict[str, object]] = []
    policy = str(cfg["local_predictor_policy"])
    layer_ids = [int(x) for x in cfg["local_predictor_layers"]]
    t_max = int(cfg["local_predictor_t_max"])
    limit_iters = int(cfg["local_predictor_limit_iters"])

    for seed in sorted(models_by_seed.keys()):
        if policy not in models_by_seed[seed]:
            continue
        model = base.unwrap(models_by_seed[seed][policy])
        for layer_id in layer_ids:
            if layer_id < 0 or layer_id >= len(model.layers):
                continue
            A0 = model.layers[layer_id].mixer._positive().detach().to(dtype=torch.float64, device="cpu")
            rows, rho_star = local_predictor_rows(A0, t_max=t_max, limit_iters=limit_iters)
            for row in rows:
                row.update({"seed": int(seed), "policy": policy, "layer": int(layer_id)})
            rows_all.extend(rows)
            print(f"seed={seed} policy={policy} layer={layer_id} rho_star={rho_star:.4g}")

            df = pd.DataFrame(rows)
            fig, ax = plt.subplots(figsize=(5.8, 3.8))
            x = df["t"].to_numpy()
            for col, label, style in [
                ("measured_eps_c", "measured", "-"),
                ("kappa_bound", r"$\kappa$-only bound", "--"),
                ("matrix_adaptive_bound", r"$\Delta(A)$ bound", "--"),
                ("local_first_order", "local first-order", "-."),
                ("local_second_order", "second-order", ":"),
            ]:
                y = df[col].clip(lower=1e-16).to_numpy()
                ax.plot(x, y, style, label=label)
            ax.set_yscale("log")
            ax.set_xlabel("full Sinkhorn steps t")
            ax.set_ylabel(r"$\varepsilon_c^{(2t)}$")
            ax.set_title(
                rf"Seed {seed}, layer {layer_id}, $\rho_\ast={rho_star:.3f}$"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"local_predictor_seed{seed}_layer{layer_id}.png"), dpi=240, bbox_inches="tight")
            plt.close(fig)

    write_csv(os.path.join(logs_dir, "local_predictor_curves.csv"), rows_all)
    return rows_all


def write_kappa_note(cfg: Dict[str, object], ablation_rows: List[Dict[str, object]]) -> None:
    logs_dir, _plots_dir = ensure_dirs(str(cfg["out_dir"]))
    df = pd.DataFrame(ablation_rows)
    final = df[df["epoch"] == df["epoch"].max()] if not df.empty else df
    sink = final[final["policy"].isin(["uniform", "adaptive", "cert_penalty"])]
    kappa_med = float(sink["kappa_median"].median()) if not sink.empty else float("nan")
    text = f"""# Observed-kappa interpretation

The observed dynamic range in the CIFAR stress runs is a realized diagnostic of the
trained positive matrices after the chosen logit scale, clipping, and temperature. It
is not the small illustrative kappa value used in the conservative lookup table.

Final available median kappa_obs across Sinkhorn policies in this run: {kappa_med:.6g}.

There is no contradiction if kappa_obs is much larger than the lookup-table range.
The lookup table is a conservative design example; the actual controller records the
measured marginal error epsilon_c, the product certificate, and the exact propagated
backward gain. Large observed kappa mainly says that the kappa-only worst-case Hilbert
bound will be conservative, so matrix-adaptive and measured diagnostics are the right
quantities to report next to the design rule.
"""
    with open(os.path.join(logs_dir, "observed_kappa_note.md"), "w") as f:
        f.write(text)


def zip_outputs(out_dir: str) -> str:
    zip_root = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.dirname(out_dir)
    zip_path = os.path.join(zip_root, "paper-extra-experiments.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                path = os.path.join(root, name)
                zf.write(path, os.path.relpath(path, os.path.dirname(out_dir)))
    return zip_path


def run_all(cfg: Dict[str, object]) -> None:
    cfg = dict(cfg)
    out_dir = str(cfg["out_dir"])
    logs_dir, _plots_dir = ensure_dirs(out_dir)
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}, n_gpus={torch.cuda.device_count() if device.type == 'cuda' else 0}")
    print(f"[out_dir] {out_dir}")
    print(f"[policies] {POLICIES}")

    ablation_rows, final_models_by_seed = run_focused_ablation(cfg, device)

    if bool(cfg.get("run_temperature_grid", True)):
        run_temperature_grid(cfg)

    if bool(cfg.get("run_local_predictor", True)):
        run_local_predictor_export(final_models_by_seed, cfg)

    write_kappa_note(cfg, ablation_rows)
    zip_path = zip_outputs(out_dir)
    print(f"\n[outputs] {out_dir}")
    print(f"[zip] {zip_path}")
    print("[done] Extra paper experiments complete.")


if __name__ == "__main__":
    run_all(CONFIG)
