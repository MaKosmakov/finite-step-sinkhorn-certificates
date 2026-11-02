#!/usr/bin/env python3
"""Tiny CIFAR-10 transformer relevance demo for finite-step Sinkhorn certificates.

This is a deliberately small diagnostic, not a benchmark.  It trains the same
tiny transformer-style CIFAR classifier twice:

  1. row-only softmax stream mixers;
  2. column-then-row Sinkhorn stream mixers with an adaptive log-budget target.

The point is to show that the certificate is operational in a real trainable
module: row-only normalization can produce a large realized backward mixer gain,
whereas the Sinkhorn budget controller keeps the depth-wise mixer product near
one while training under the same data/model setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cifar10_subset(
    data_dir: Path,
    *,
    train_size: int,
    test_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a deterministic CIFAR-10 subset from the python batch files."""
    if not (data_dir / "data_batch_1").exists() and (data_dir / "cifar-10-batches-py" / "data_batch_1").exists():
        data_dir = data_dir / "cifar-10-batches-py"

    def read_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        x = obj["data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
        y = np.array(obj["labels"], dtype="int64")
        return x, y

    train_xs, train_ys = [], []
    for i in range(1, 6):
        x, y = read_batch(data_dir / f"data_batch_{i}")
        train_xs.append(x)
        train_ys.append(y)
    x_train = np.concatenate(train_xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)
    x_test, y_test = read_batch(data_dir / "test_batch")

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(x_train))[:train_size]
    test_idx = rng.permutation(len(x_test))[:test_size]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    x_test, y_test = x_test[test_idx], y_test[test_idx]

    mean = np.array([0.4914, 0.4822, 0.4465], dtype="float32").reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype="float32").reshape(1, 3, 1, 1)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return (
        torch.tensor(x_train),
        torch.tensor(y_train),
        torch.tensor(x_test),
        torch.tensor(y_test),
    )


def iterate_minibatches(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator()
    g.manual_seed(seed + 1009 * epoch)
    idx = torch.randperm(x.shape[0], generator=g)
    for start in range(0, x.shape[0], batch_size):
        sl = idx[start : start + batch_size]
        yield x[sl], y[sl]


def finite_sinkhorn(
    logits: torch.Tensor,
    *,
    beta: float,
    clip_logits: float,
    t_max: int,
    eps_target: float,
    safe_eps: float = 1e-12,
) -> Tuple[torch.Tensor, int, float, float, bool]:
    z = (beta * logits).clamp(-clip_logits, clip_logits)
    x = torch.exp(z)
    t_used = 0
    cap_failed = True
    eps_c_val = float("inf")
    eps_r_val = float("inf")
    for t in range(int(t_max)):
        x = x / x.sum(dim=0, keepdim=True).clamp_min(safe_eps)
        x = x / x.sum(dim=1, keepdim=True).clamp_min(safe_eps)
        t_used = t + 1
        with torch.no_grad():
            eps_r_val = float((x.sum(dim=1) - 1.0).abs().max().detach().cpu())
            eps_c_val = float((x.sum(dim=0) - 1.0).abs().max().detach().cpu())
            if eps_c_val <= eps_target:
                cap_failed = False
                break
    return x, t_used, eps_r_val, eps_c_val, cap_failed


class StreamMixer(nn.Module):
    def __init__(
        self,
        n_streams: int,
        *,
        mode: str,
        beta: float,
        clip_logits: float,
        t_max: int,
        init_noise: float,
        collapse_strength: float,
    ) -> None:
        super().__init__()
        self.n_streams = int(n_streams)
        self.mode = str(mode)
        self.beta = float(beta)
        self.clip_logits = float(clip_logits)
        self.t_max = int(t_max)
        init = init_noise * torch.randn(n_streams, n_streams)
        init[:, 0] += float(collapse_strength)
        self.logits = nn.Parameter(init)
        self.last_t = 0
        self.last_eps_r = float("nan")
        self.last_eps_c = float("nan")
        self.last_cap_failed = False

    def matrix(self, eps_target: float | None = None) -> torch.Tensor:
        if self.mode == "row_softmax":
            z = (self.beta * self.logits).clamp(-self.clip_logits, self.clip_logits)
            h = F.softmax(z, dim=1)
            with torch.no_grad():
                self.last_t = 1
                self.last_eps_r = float((h.sum(dim=1) - 1.0).abs().max().detach().cpu())
                self.last_eps_c = float((h.sum(dim=0) - 1.0).abs().max().detach().cpu())
                self.last_cap_failed = False
            return h
        if eps_target is None:
            eps_target = 5e-4
        h, t_used, eps_r, eps_c, cap_failed = finite_sinkhorn(
            self.logits,
            beta=self.beta,
            clip_logits=self.clip_logits,
            t_max=self.t_max,
            eps_target=float(eps_target),
        )
        self.last_t = int(t_used)
        self.last_eps_r = float(eps_r)
        self.last_eps_c = float(eps_c)
        self.last_cap_failed = bool(cap_failed)
        return h


class TinyTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        n_streams: int,
        dim: int,
        heads: int,
        mlp_ratio: int,
        mixer_mode: str,
        beta: float,
        clip_logits: float,
        sinkhorn_t_max: int,
        mixer_init_noise: float,
        collapse_strength: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.mixer = StreamMixer(
            n_streams,
            mode=mixer_mode,
            beta=beta,
            clip_logits=clip_logits,
            t_max=sinkhorn_t_max,
            init_noise=mixer_init_noise,
            collapse_strength=collapse_strength,
        )

    def forward(self, x: torch.Tensor, eps_target: float | None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, t, d = x.shape
        h = x.reshape(b * s, t, d)
        y = self.ln1(h)
        y, _ = self.attn(y, y, y, need_weights=False)
        h = h + y
        h = h + self.mlp(self.ln2(h))
        h = h.reshape(b, s, t, d)
        mix = self.mixer.matrix(eps_target)
        h = torch.einsum("ij,bjtd->bitd", mix, h)
        return h, mix


class TinyStreamTransformer(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        n_streams: int,
        dim: int,
        heads: int,
        patch: int,
        mixer_mode: str,
        beta: float,
        clip_logits: float,
        sinkhorn_t_max: int,
        sinkhorn_eps: float,
        mixer_init_noise: float,
        collapse_strength: float,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.n_streams = int(n_streams)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.log_budget = self.depth * math.log1p(self.sinkhorn_eps)
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_tokens = (32 // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, 1, n_tokens, dim))
        self.stream_embed = nn.Parameter(0.02 * torch.randn(1, n_streams, 1, dim))
        self.blocks = nn.ModuleList(
            [
                TinyTransformerBlock(
                    n_streams=n_streams,
                    dim=dim,
                    heads=heads,
                    mlp_ratio=2,
                    mixer_mode=mixer_mode,
                    beta=beta,
                    clip_logits=clip_logits,
                    sinkhorn_t_max=sinkhorn_t_max,
                    mixer_init_noise=mixer_init_noise,
                    collapse_strength=collapse_strength,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x).flatten(2).transpose(1, 2)
        h = h.unsqueeze(1).repeat(1, self.n_streams, 1, 1)
        h = h + self.pos + self.stream_embed

        used_log_budget = 0.0
        for i, block in enumerate(self.blocks):
            eps_target = None
            if block.mixer.mode == "sinkhorn":
                remaining = max(self.depth - i, 1)
                eps_target = math.expm1(max(self.log_budget - used_log_budget, 0.0) / remaining)
            h, _ = block(h, eps_target)
            if block.mixer.mode == "sinkhorn":
                used_log_budget += math.log1p(max(block.mixer.last_eps_c, 0.0))

        h = self.norm(h).mean(dim=(1, 2))
        return self.head(h)

    @torch.no_grad()
    def mixer_diagnostics(self) -> Dict[str, float]:
        matrices: List[torch.Tensor] = []
        eps_cs: List[float] = []
        eps_rs: List[float] = []
        steps: List[float] = []
        cap_fails: List[float] = []

        used_log_budget = 0.0
        for i, block in enumerate(self.blocks):
            eps_target = None
            if block.mixer.mode == "sinkhorn":
                remaining = max(self.depth - i, 1)
                eps_target = math.expm1(max(self.log_budget - used_log_budget, 0.0) / remaining)
            h = block.mixer.matrix(eps_target)
            matrices.append(h.detach().cpu())
            eps_cs.append(block.mixer.last_eps_c)
            eps_rs.append(block.mixer.last_eps_r)
            steps.append(block.mixer.last_t)
            cap_fails.append(1.0 if block.mixer.last_cap_failed else 0.0)
            if block.mixer.mode == "sinkhorn":
                used_log_budget += math.log1p(max(block.mixer.last_eps_c, 0.0))

        w = torch.ones(self.n_streams, dtype=torch.float64)
        for h in reversed(matrices):
            w = h.to(torch.float64).T @ w
        exact_t1 = float(w.max().item())
        log_cert = float(np.sum(np.log1p(np.maximum(eps_cs, 0.0))))
        cert = float(math.exp(log_cert)) if log_cert < 700 else float("inf")
        return {
            "exact_T1": exact_t1,
            "product_cert": cert,
            "max_eps_c": float(np.max(eps_cs)),
            "max_eps_r": float(np.max(eps_rs)),
            "avg_steps": float(np.mean(steps)),
            "cap_failure_rate": float(np.mean(cap_fails)),
            "max_column_sum": float(max(float(h.sum(dim=0).max()) for h in matrices)),
        }

    @torch.no_grad()
    def mixer_stack(self) -> np.ndarray:
        """Return final/evaluation residual mixer matrices with shape (L,n,n)."""
        matrices: List[torch.Tensor] = []
        used_log_budget = 0.0
        for i, block in enumerate(self.blocks):
            eps_target = None
            if block.mixer.mode == "sinkhorn":
                remaining = max(self.depth - i, 1)
                eps_target = math.expm1(max(self.log_budget - used_log_budget, 0.0) / remaining)
            h = block.mixer.matrix(eps_target)
            matrices.append(h.detach().cpu())
            if block.mixer.mode == "sinkhorn":
                used_log_budget += math.log1p(max(block.mixer.last_eps_c, 0.0))
        return torch.stack(matrices).to(torch.float64).numpy()


def make_model(args: argparse.Namespace, mode: str) -> TinyStreamTransformer:
    return TinyStreamTransformer(
        depth=args.depth,
        n_streams=args.n_streams,
        dim=args.dim,
        heads=args.heads,
        patch=args.patch,
        mixer_mode=mode,
        beta=args.beta,
        clip_logits=args.clip_logits,
        sinkhorn_t_max=args.sinkhorn_t_max,
        sinkhorn_eps=args.sinkhorn_eps,
        mixer_init_noise=args.mixer_init_noise,
        collapse_strength=args.collapse_strength,
    )


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        v = float(p.grad.detach().float().norm().cpu())
        total += v * v
    return math.sqrt(total)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, batch_size: int, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.cpu()) * xb.shape[0]
        total_correct += int((logits.argmax(dim=1) == yb).sum().cpu())
        total += xb.shape[0]
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one_policy(
    args: argparse.Namespace,
    *,
    mode: str,
    label: str,
    initial_state: Dict[str, torch.Tensor],
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> List[Dict[str, float | int | str]]:
    x_train, y_train, x_test, y_test = data
    model = make_model(args, mode).to(device)
    model.load_state_dict(initial_state)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logs: List[Dict[str, float | int | str]] = []
    last_diag: Dict[str, float] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.time()
        losses: List[float] = []
        grads: List[float] = []
        nonfinite = False
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size=args.batch_size, seed=args.seed, epoch=epoch):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                nonfinite = True
                break
            loss.backward()
            gnorm = grad_norm(model.parameters())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            grads.append(float(gnorm))

        eval_loss, eval_acc = evaluate(model, x_test, y_test, batch_size=args.batch_size, device=device)
        diag = model.mixer_diagnostics()
        last_diag = diag
        row = {
            "seed": int(args.seed),
            "policy": label,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "test_loss": float(eval_loss),
            "test_acc": float(eval_acc),
            "grad_norm_mean": float(np.mean(grads)) if grads else float("nan"),
            "grad_norm_max": float(np.max(grads)) if grads else float("nan"),
            "epoch_time_sec": float(time.time() - tic),
            "nonfinite": int(nonfinite),
            **diag,
        }
        logs.append(row)
        print(
            f"seed {args.seed:02d} {label:16s} epoch {epoch:02d}: acc={row['test_acc']:.3f} "
            f"loss={row['test_loss']:.3f} T1={row['exact_T1']:.3f} "
            f"cert={row['product_cert']:.3g} max_eps_c={row['max_eps_c']:.3g} "
            f"grad_max={row['grad_norm_max']:.2f}"
        )
        if nonfinite:
            break
    if getattr(args, "save_mixers", False) and logs:
        save_mixer_stack(args, model, label, logs[-1], last_diag or {})
    return logs


def safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def save_mixer_stack(
    args: argparse.Namespace,
    model: TinyStreamTransformer,
    label: str,
    final_row: Dict[str, float | int | str],
    diag: Dict[str, float],
) -> None:
    mixer_dir = args.mixer_out_dir or (args.out_dir / "mixers")
    mixer_dir.mkdir(parents=True, exist_ok=True)
    policy_name = safe_name(label)
    seed = int(final_row["seed"])
    epoch = int(final_row["epoch"])
    path = mixer_dir / f"seed{seed:02d}_{policy_name}_epoch{epoch:02d}_mixers.npz"
    h_res = model.mixer_stack()
    np.savez_compressed(
        path,
        H_res=h_res,
        policy=np.array(label),
        seed=np.array(seed, dtype=np.int64),
        epoch=np.array(epoch, dtype=np.int64),
        L=np.array(h_res.shape[0], dtype=np.int64),
        n=np.array(h_res.shape[1], dtype=np.int64),
        exact_T1=np.array(float(diag.get("exact_T1", np.nan)), dtype=np.float64),
        product_cert=np.array(float(diag.get("product_cert", np.nan)), dtype=np.float64),
        max_eps_c=np.array(float(diag.get("max_eps_c", np.nan)), dtype=np.float64),
        note=np.array("final/evaluation per-layer residual mixer stack with shape (L,n,n)"),
    )
    print(f"  saved mixers: {path}")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float, ndigits: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "---"
    if abs(float(x)) >= 1e4 or (0 < abs(float(x)) < 1e-3):
        return f"{float(x):.2e}"
    return f"{float(x):.{ndigits}f}"


def mean_std(rows: List[Dict[str, object]], key: str) -> Tuple[float, float]:
    vals = np.array([float(r[key]) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1))


def fmt_pm(rows: List[Dict[str, object]], key: str, ndigits: int = 3) -> str:
    mu, sd = mean_std(rows, key)
    if len({int(r["seed"]) for r in rows}) <= 1:
        return fmt(mu, ndigits)
    return f"{fmt(mu, ndigits)}$\\pm${fmt(sd, ndigits)}"


def write_latex_table(path: Path, final_rows: List[Dict[str, object]]) -> None:
    order = ["row-only softmax", "Sinkhorn budget"]
    seeds = sorted({int(r["seed"]) for r in final_rows})
    seed_text = ", ".join(str(s) for s in seeds)
    tex = [
        r"\begin{center}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Policy & Acc. & Loss & Exact $\|T_L\|_1$ & Product cert. \\",
        r"\midrule",
    ]
    for policy in order:
        policy_rows = [r for r in final_rows if str(r["policy"]) == policy]
        tex.append(
            f"{policy} & "
            f"{fmt_pm(policy_rows, 'test_acc')} & "
            f"{fmt_pm(policy_rows, 'test_loss')} & "
            f"{fmt_pm(policy_rows, 'exact_T1')} & "
            f"{fmt_pm(policy_rows, 'product_cert')} \\\\"
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\captionof{table}{Matched short-horizon CIFAR-10 transformer diagnostic. Both rows use the same",
            r"initial weights and minibatches within each seed; only the stream-mixer normalization changes.",
            r"The row-only mixer is row-stochastic but not column-controlled, while the Sinkhorn",
            rf"row uses the adaptive log-budget controller. Statistics are mean$\pm$std over seeds {seed_text}.}}",
            r"\label{tab:tiny_transformer_relevance}",
            r"\end{center}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tex) + "\n")


def make_plot(path: Path, logs: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    policies = ["row-only softmax", "Sinkhorn budget"]
    colors = {"row-only softmax": "#d95f02", "Sinkhorn budget": "#1b9e77"}

    def draw_panel(ax: plt.Axes, metric: str, ylabel: str) -> None:
        for policy in policies:
            rows = [r for r in logs if r["policy"] == policy]
            epochs = sorted({int(r["epoch"]) for r in rows})
            xs: List[int] = []
            means: List[float] = []
            stds: List[float] = []
            for epoch in epochs:
                epoch_rows = [r for r in rows if int(r["epoch"]) == epoch]
                mu, sd = mean_std(epoch_rows, metric)
                xs.append(epoch)
                means.append(mu)
                stds.append(sd)
            ax.plot(xs, means, marker="o", linewidth=2.0, label=policy, color=colors[policy])
            if len({int(r["seed"]) for r in rows}) > 1:
                lo = np.array(means) - np.array(stds)
                hi = np.array(means) + np.array(stds)
                if metric in {"exact_T1", "max_eps_c", "grad_norm_max"}:
                    lo = np.maximum(lo, 1e-12)
                ax.fill_between(xs, lo, hi, alpha=0.15, color=colors[policy], linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if metric in {"exact_T1", "max_eps_c", "grad_norm_max"}:
            ax.set_yscale("log")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    draw_panel(axes[0], "exact_T1", r"exact $\|T_L\|_1$")
    draw_panel(axes[1], "test_acc", "test accuracy")
    axes[0].axhline(1.0, linestyle="--", linewidth=1.0, color="0.4")
    axes[0].axhline(8.0, linestyle=":", linewidth=1.0, color="0.4")
    axes[0].text(0.03, 0.86, "stream ceiling n=8", transform=axes[0].transAxes, fontsize=8)
    axes[0].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    extra_path = path.with_name(f"{path.stem}_extra.png")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    draw_panel(axes[0], "max_eps_c", r"max column error $\varepsilon_c$")
    draw_panel(axes[1], "grad_norm_max", "max gradient norm")
    axes[0].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(extra_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("results-stressed/data/cifar-10-batches-py"))
    parser.add_argument("--out_dir", type=Path, default=Path("relevance-demo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train_size", type=int, default=768)
    parser.add_argument("--test_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--n_streams", type=int, default=8)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--clip_logits", type=float, default=8.0)
    parser.add_argument("--sinkhorn_t_max", type=int, default=40)
    parser.add_argument("--sinkhorn_eps", type=float, default=5e-4)
    parser.add_argument("--mixer_init_noise", type=float, default=0.05)
    parser.add_argument("--collapse_strength", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--save_mixers",
        action="store_true",
        help="Save final/evaluation per-layer residual mixer stacks as compressed .npz files.",
    )
    parser.add_argument(
        "--mixer_out_dir",
        type=Path,
        default=None,
        help="Optional output directory for saved mixer stacks. Defaults to OUT_DIR/mixers.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = list(args.seeds) if args.seeds is not None else [int(args.seed)]
    set_seed(seeds[0])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    if args.save_mixers:
        (args.mixer_out_dir or (args.out_dir / "mixers")).mkdir(parents=True, exist_ok=True)

    all_logs: List[Dict[str, object]] = []
    final_rows: List[Dict[str, object]] = []
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = int(seed)
        set_seed(int(seed))
        data = load_cifar10_subset(
            run_args.data_dir,
            train_size=run_args.train_size,
            test_size=run_args.test_size,
            seed=int(seed),
        )
        base_model = make_model(run_args, "row_softmax")
        initial_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        for mode, label in [("row_softmax", "row-only softmax"), ("sinkhorn", "Sinkhorn budget")]:
            set_seed(int(seed))
            logs = train_one_policy(
                run_args,
                mode=mode,
                label=label,
                initial_state=initial_state,
                data=data,
                device=device,
            )
            all_logs.extend(logs)
            final_rows.append(logs[-1])

    write_csv(args.out_dir / "logs" / "tiny_transformer_relevance_epoch_logs.csv", all_logs)
    write_csv(args.out_dir / "logs" / "tiny_transformer_relevance_summary.csv", final_rows)
    write_latex_table(args.out_dir / "logs" / "tiny_transformer_relevance_table.tex", final_rows)
    make_plot(args.out_dir / "plots" / "tiny_transformer_relevance.png", all_logs)
    with (args.out_dir / "logs" / "config.json").open("w") as f:
        cfg = vars(args).copy()
        cfg["resolved_seeds"] = seeds
        json.dump(cfg, f, indent=2, default=str)

    print("\nWrote:")
    print(f"  {args.out_dir / 'logs' / 'tiny_transformer_relevance_epoch_logs.csv'}")
    print(f"  {args.out_dir / 'logs' / 'tiny_transformer_relevance_summary.csv'}")
    print(f"  {args.out_dir / 'logs' / 'tiny_transformer_relevance_table.tex'}")
    print(f"  {args.out_dir / 'plots' / 'tiny_transformer_relevance.png'}")
    if args.save_mixers:
        print(f"  {args.mixer_out_dir or (args.out_dir / 'mixers')}")


if __name__ == "__main__":
    main()
