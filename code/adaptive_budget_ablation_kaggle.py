
# Adaptive-budget Sinkhorn ablation for Kaggle GPU.
# Runs three Sinkhorn stopping policies with the same architecture, seeds, data order,
# and initialization: fixed cap, uniform tolerance, adaptive global log-budget.

from __future__ import annotations

import os, math, time, random, tarfile, csv, json, zipfile, copy
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

import torchvision
import torchvision.transforms as T
from cifar_data import load_cifar10

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    display = print


# =========================
# Kaggle-editable settings
# =========================
CONFIG = {
    # Reported run: [0, 1, 2]. Smoke test: [0].
    'seeds': [0, 1, 2],
    'epochs': 5,

    # CIFAR / runtime.
    'batch_size': 256,
    'num_workers': 4,
    'amp': True,
    'data_parallel': False,
    'data_root': '/kaggle/input/cifar10-python',
    'out_dir': '/kaggle/working/adaptive-budget-ablation',

    # Optional runtime limits for smoke tests. Keep None for reported runs.
    'max_train_batches': None,
    'max_eval_batches': None,

    # Stress profile used in the paper.
    'depth': 128,
    'n_streams': 8,
    'd': 128,
    'beta_sinkhorn': 3.0,
    'init_scale': 2.5,
    'sinkhorn_t_max': 80,
    'sinkhorn_eps_tol': 5e-4,
    'sinkhorn_clip_logits': 10.0,
    'lr': 0.02,
    'momentum': 0.9,
    'weight_decay': 5e-4,
}

POLICIES = ['fixed', 'uniform', 'adaptive']
POLICY_LABELS = {
    'fixed': r'Fixed $t_{\max}=80$',
    'uniform': 'Uniform tolerance',
    'adaptive': 'Adaptive log-budget',
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, data_source, base_seed: int, shuffle: bool = True) -> None:
        self.data_source = data_source
        self.base_seed = int(base_seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        n = len(self.data_source)
        if not self.shuffle:
            return iter(range(n))
        gen = torch.Generator()
        gen.manual_seed(self.base_seed + self.epoch)
        return iter(torch.randperm(n, generator=gen).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


@torch.no_grad()
def marginal_errors(A: torch.Tensor) -> Tuple[float, float]:
    n = A.shape[0]
    ones = torch.ones(n, device=A.device, dtype=A.dtype)
    er = (A.sum(dim=1) - ones).abs().max().item()
    ec = (A.sum(dim=0) - ones).abs().max().item()
    return float(er), float(ec)


def finite_sinkhorn_policy(
    M: torch.Tensor,
    *,
    policy: str,
    t_max: int,
    eps_target: Optional[float],
    safe_eps: float = 1e-20,
) -> Tuple[torch.Tensor, int, float, float, float, bool]:
    """Column-then-row Sinkhorn. Early policies stop on column error only."""
    X = M
    target = float('nan') if eps_target is None else float(eps_target)
    t_used = 0

    if policy == 'fixed':
        for t in range(int(t_max)):
            X = X / X.sum(dim=0, keepdim=True).clamp_min(safe_eps)
            X = X / X.sum(dim=1, keepdim=True).clamp_min(safe_eps)
            t_used = t + 1
        er, ec = marginal_errors(X)
        return X, int(t_used), er, ec, target, True

    if not math.isfinite(target):
        raise ValueError('uniform/adaptive policies require a finite eps_target')
    target = max(target, 0.0)

    er0, ec0 = marginal_errors(X)
    if ec0 <= target:
        return X, 0, er0, ec0, target, True

    er = er0
    ec = ec0
    for t in range(int(t_max)):
        X = X / X.sum(dim=0, keepdim=True).clamp_min(safe_eps)
        X = X / X.sum(dim=1, keepdim=True).clamp_min(safe_eps)
        t_used = t + 1
        er, ec = marginal_errors(X)
        if ec <= target:
            break

    hit = bool(ec <= target)
    return X, int(t_used), float(er), float(ec), target, hit


class SinkhornMixer(nn.Module):
    def __init__(self, n: int, cfg: Dict[str, object], policy: str):
        super().__init__()
        self.n = int(n)
        self.policy = str(policy)
        self.beta = float(cfg['beta_sinkhorn'])
        self.t_max = int(cfg['sinkhorn_t_max'])
        self.eps_tol = float(cfg['sinkhorn_eps_tol'])
        self.clip_logits = float(cfg['sinkhorn_clip_logits'])
        self.W = nn.Parameter(torch.empty(n, n))
        nn.init.normal_(self.W, mean=0.0, std=float(cfg['init_scale']) / math.sqrt(n))

        self.last_t_stop = 0
        self.last_eps_r = float('nan')
        self.last_eps_c = float('nan')
        self.last_eps_target = float('nan')
        self.last_hit = True

    def _positive(self) -> torch.Tensor:
        z = self.beta * self.W
        if self.clip_logits > 0:
            z = z.clamp(-self.clip_logits, self.clip_logits)
        return torch.exp(z)

    def forward(self, eps_target: Optional[float] = None) -> torch.Tensor:
        target = None
        if self.policy == 'uniform':
            target = self.eps_tol
        elif self.policy == 'adaptive':
            target = self.eps_tol if eps_target is None else float(eps_target)

        A, t_used, er, ec, target_used, hit = finite_sinkhorn_policy(
            self._positive(),
            policy=self.policy,
            t_max=self.t_max,
            eps_target=target,
        )
        self.last_t_stop = int(t_used)
        self.last_eps_r = float(er)
        self.last_eps_c = float(ec)
        self.last_eps_target = float(target_used)
        self.last_hit = bool(hit)
        return A

    @torch.no_grad()
    def fixed_matrix(self) -> torch.Tensor:
        A, *_ = finite_sinkhorn_policy(
            self._positive(),
            policy='fixed',
            t_max=self.t_max,
            eps_target=None,
        )
        return A


class StreamBlock(nn.Module):
    def __init__(self, d: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = int(d * mlp_ratio)
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class MixerLayer(nn.Module):
    def __init__(self, n_streams: int, d: int, mixer: SinkhornMixer):
        super().__init__()
        self.block = StreamBlock(d=d)
        self.mixer = mixer

    def forward(self, x: torch.Tensor, eps_target: Optional[float] = None) -> torch.Tensor:
        x = self.block(x)
        A = self.mixer(eps_target=eps_target)
        return torch.einsum('ij,bjd->bid', A, x)


class CIFARStem(nn.Module):
    def __init__(self, d: int, n_streams: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(256 * 8 * 8, n_streams * d)
        self.n_streams = int(n_streams)
        self.d = int(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.proj(h).view(h.shape[0], self.n_streams, self.d)


class MixerNet(nn.Module):
    def __init__(self, cfg: Dict[str, object], policy: str):
        super().__init__()
        self.policy = str(policy)
        self.depth = int(cfg['depth'])
        self.eps_tol = float(cfg['sinkhorn_eps_tol'])
        self.log_budget = self.depth * math.log1p(self.eps_tol)
        n_streams = int(cfg['n_streams'])
        d = int(cfg['d'])
        self.stem = CIFARStem(d=d, n_streams=n_streams)
        self.layers = nn.ModuleList([
            MixerLayer(n_streams=n_streams, d=d, mixer=SinkhornMixer(n_streams, cfg, policy))
            for _ in range(self.depth)
        ])
        self.head_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        used_log_budget = 0.0
        for ell, layer in enumerate(self.layers):
            eps_target = None
            if self.policy == 'adaptive':
                remaining_layers = self.depth - ell
                remaining_budget = max(self.log_budget - used_log_budget, 0.0)
                eps_target = math.expm1(remaining_budget / max(remaining_layers, 1))
            x = layer(x, eps_target=eps_target)
            if self.policy == 'adaptive':
                eps_c = max(float(layer.mixer.last_eps_c), 0.0)
                used_log_budget += math.log1p(eps_c)
        x = self.head_ln(x).mean(dim=1)
        return self.head(x)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def kappa_obs(A: torch.Tensor) -> float:
    tiny = torch.finfo(A.dtype).tiny
    return float(A.max().item() / A.min().clamp_min(tiny).item())


@torch.no_grad()
def stack_mixer_metrics(mats: List[torch.Tensor]) -> Dict[str, float]:
    if not mats:
        return {'max_eps_c': float('nan'), 'max_eps_r': float('nan'), 'cert_bound': float('nan'), 'log_cert': float('nan'), 'exact_T1': float('nan')}
    mats64 = [A.detach().to(dtype=torch.float64, device='cpu') for A in mats]
    n = mats64[0].shape[0]
    ones = torch.ones(n, dtype=torch.float64)
    eps_c = [float((A.sum(dim=0) - ones).abs().max().item()) for A in mats64]
    eps_r = [float((A.sum(dim=1) - ones).abs().max().item()) for A in mats64]
    log_cert = float(sum(math.log1p(max(e, 0.0)) for e in eps_c))
    cert_bound = float(math.exp(log_cert)) if log_cert < 700 else float('inf')
    w = torch.ones(n, dtype=torch.float64)
    for A in reversed(mats64):
        w = A.T @ w
    return {
        'max_eps_c': float(max(eps_c)),
        'max_eps_r': float(max(eps_r)),
        'cert_bound': cert_bound,
        'log_cert': log_cert,
        'exact_T1': float(w.abs().max().item()),
    }


@torch.no_grad()
def collect_policy_stats(model: nn.Module) -> Dict[str, float]:
    m = unwrap(model)
    mats: List[torch.Tensor] = []
    t_stops: List[int] = []
    eps_cs: List[float] = []
    eps_rs: List[float] = []
    targets: List[float] = []
    hits: List[bool] = []
    kappas: List[float] = []

    used_log_budget = 0.0
    for ell, layer in enumerate(m.layers):
        eps_target = None
        if m.policy == 'adaptive':
            remaining_layers = m.depth - ell
            remaining_budget = max(m.log_budget - used_log_budget, 0.0)
            eps_target = math.expm1(remaining_budget / max(remaining_layers, 1))
        A = layer.mixer(eps_target=eps_target)
        if m.policy == 'adaptive':
            used_log_budget += math.log1p(max(float(layer.mixer.last_eps_c), 0.0))
        mats.append(A)
        t_stops.append(int(layer.mixer.last_t_stop))
        eps_rs.append(float(layer.mixer.last_eps_r))
        eps_cs.append(float(layer.mixer.last_eps_c))
        targets.append(float(layer.mixer.last_eps_target))
        hits.append(bool(layer.mixer.last_hit))
        kappas.append(kappa_obs(A))

    stack = stack_mixer_metrics(mats)
    finite_targets = np.array([x for x in targets if math.isfinite(x)], dtype=float)
    return {
        'avg_steps': float(np.mean(t_stops)),
        'median_steps': float(np.median(t_stops)),
        'max_eps_c': float(np.max(eps_cs)),
        'max_eps_r': float(np.max(eps_rs)),
        'target_eps_median': float(np.median(finite_targets)) if finite_targets.size else float('nan'),
        'target_eps_min': float(np.min(finite_targets)) if finite_targets.size else float('nan'),
        'target_eps_max': float(np.max(finite_targets)) if finite_targets.size else float('nan'),
        'cap_failure_rate': 0.0 if m.policy == 'fixed' else float(1.0 - np.mean(hits)),
        'kappa_median': float(np.median(kappas)),
        'budget_B1': float(math.exp(m.log_budget)),
        'budget_log': float(m.log_budget),
        'used_log_budget': float(stack['log_cert']),
        **stack,
    }


def get_loaders(cfg: Dict[str, object], seed: int):
    root = str(cfg['data_root'])
    tr = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    te = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = load_cifar10(root, train=True, transform=tr)
    test_ds = load_cifar10(root, train=False, transform=te)
    sampler = EpochShuffleSampler(train_ds, base_seed=seed, shuffle=True)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg['batch_size']),
        sampler=sampler,
        num_workers=int(cfg['num_workers']),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        persistent_workers=(int(cfg['num_workers']) > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg['batch_size']),
        shuffle=False,
        num_workers=int(cfg['num_workers']),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        persistent_workers=(int(cfg['num_workers']) > 0),
    )
    return train_loader, test_loader, sampler


def make_policy_models(cfg: Dict[str, object], device: torch.device) -> Dict[str, nn.Module]:
    base = MixerNet(cfg, policy='fixed').to(device)
    base_state = copy.deepcopy(base.state_dict())
    models = {'fixed': base}
    for policy in ['uniform', 'adaptive']:
        model = MixerNet(cfg, policy=policy).to(device)
        model.load_state_dict(base_state)
        models[policy] = model
    return models


@torch.no_grad()
def acc_value(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def train_one_epoch(models, optimizers, scalers, loader, device, cfg) -> Dict[str, Tuple[float, float, float]]:
    for model in models.values():
        model.train(True)
    totals = {p: {'loss': 0.0, 'acc': 0.0, 'n': 0, 'time': 0.0} for p in POLICIES}
    amp_enabled = bool(cfg['amp']) and device.type == 'cuda'
    max_batches = cfg.get('max_train_batches')

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
                loss = F.cross_entropy(logits, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            totals[policy]['time'] += time.perf_counter() - t0
            totals[policy]['loss'] += float(loss.item()) * bs
            totals[policy]['acc'] += acc_value(logits.detach(), y) * bs
            totals[policy]['n'] += bs
    return {p: (v['loss'] / v['n'], v['acc'] / v['n'], v['time']) for p, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> Tuple[float, float, float]:
    model.train(False)
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    amp_enabled = bool(cfg['amp']) and device.type == 'cuda'
    max_batches = cfg.get('max_eval_batches')
    t0 = time.perf_counter()
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        bs = int(x.shape[0])
        total_loss += float(loss.item()) * bs
        total_acc += acc_value(logits, y) * bs
        total_n += bs
    return total_loss / total_n, total_acc / total_n, time.perf_counter() - t0


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_pm(values: pd.Series, precision: int = 3) -> str:
    vals = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return '---'
    if vals.size == 1:
        return f'{vals.mean():.{precision}g}'
    return f'{vals.mean():.{precision}g}$\\pm${vals.std(ddof=0):.{precision}g}'


def write_summary_outputs(rows: List[Dict[str, object]], cfg: Dict[str, object]) -> None:
    out_dir = str(cfg['out_dir'])
    logs_dir = os.path.join(out_dir, 'logs')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(logs_dir, 'adaptive_ablation_epoch_logs.csv'), index=False)

    final_epoch = int(df['epoch'].max())
    complete_seeds = []
    for seed, seed_df in df.groupby('seed'):
        seed_final = seed_df[seed_df['epoch'] == final_epoch]
        if set(seed_final['policy']) >= set(POLICIES):
            complete_seeds.append(seed)
    if complete_seeds:
        df_summary = df[df['seed'].isin(complete_seeds)].copy()
    else:
        # Fallback for partial debug runs that have not completed the first epoch.
        df_summary = df.copy()
    final = df_summary[df_summary['epoch'] == final_epoch].copy()
    summary_rows = []
    for policy in POLICIES:
        sub_final = final[final['policy'] == policy]
        sub_all = df_summary[df_summary['policy'] == policy]
        summary_rows.append({
            'policy': policy,
            'label': POLICY_LABELS[policy],
            'n_completed_seeds': len(complete_seeds),
            'test_acc_mean': sub_final['test_acc'].mean(),
            'test_acc_std': sub_final['test_acc'].std(ddof=0),
            'avg_steps_mean': sub_final['avg_steps'].mean(),
            'avg_steps_std': sub_final['avg_steps'].std(ddof=0),
            'time_per_epoch_sec_mean': sub_all['epoch_time_sec'].mean(),
            'time_per_epoch_sec_std': sub_all['epoch_time_sec'].std(ddof=0),
            'cert_bound_mean': sub_final['cert_bound'].mean(),
            'cert_bound_std': sub_final['cert_bound'].std(ddof=0),
            'exact_T1_mean': sub_final['exact_T1'].mean(),
            'exact_T1_std': sub_final['exact_T1'].std(ddof=0),
            'cap_failure_rate_mean': sub_final['cap_failure_rate'].mean(),
            'cap_failure_rate_std': sub_final['cap_failure_rate'].std(ddof=0),
            'max_eps_c_mean': sub_final['max_eps_c'].mean(),
            'max_eps_c_std': sub_final['max_eps_c'].std(ddof=0),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(logs_dir, 'adaptive_ablation_summary.csv'), index=False)

    tex = []
    tex.append(r'\begin{table}[t]')
    tex.append(r'\centering')
    tex.append(r'\small')
    tex.append(r'\resizebox{\textwidth}{!}{%')
    tex.append(r'\begin{tabular}{lcccccc}')
    tex.append(r'\toprule')
    tex.append(r'Policy')
    tex.append(r'& Test acc.')
    tex.append(r'& Avg. steps')
    tex.append(r'& Time/epoch')
    tex.append(r'& Product cert.')
    tex.append(r'& Exact $\|T_L\|_1$')
    tex.append(r'& Cap fail. \\')
    tex.append(r'\midrule')
    for policy in POLICIES:
        s = summary[summary['policy'] == policy].iloc[0]
        tex.append(str(s['label']))
        tex.append(f"& ${s['test_acc_mean']:.4f}\\pm{s['test_acc_std']:.4f}$")
        tex.append(f"& ${s['avg_steps_mean']:.2f}\\pm{s['avg_steps_std']:.2f}$")
        tex.append(f"& ${s['time_per_epoch_sec_mean']:.1f}\\pm{s['time_per_epoch_sec_std']:.1f}$s")
        tex.append(f"& ${s['cert_bound_mean']:.5f}\\pm{s['cert_bound_std']:.5f}$")
        tex.append(f"& ${s['exact_T1_mean']:.6f}\\pm{s['exact_T1_std']:.8f}$")
        tex.append(f"& ${100.0*s['cap_failure_rate_mean']:.2f}\\pm{100.0*s['cap_failure_rate_std']:.2f}\\%$ \\\\")
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'}')
    b1 = (1.0 + float(cfg['sinkhorn_eps_tol'])) ** int(cfg['depth'])
    complete_seed_text = ', '.join(str(int(s)) for s in complete_seeds)
    incomplete_seeds = sorted(set(df['seed'].unique()) - set(complete_seeds))
    tex.append(r'\caption{')
    tex.append(r'Targeted adaptive-budget ablation under the CIFAR-10 stress setting with')
    tex.append(rf'$L={int(cfg["depth"])}$, $n={int(cfg["n_streams"])}$, $d={int(cfg["d"])}$, $t_{{\max}}={int(cfg["sinkhorn_t_max"])}$, and shared budget')
    tex.append(rf'$B_1=(1+5\cdot10^{{-4}})^{{128}}={b1:.6f}$.')
    if incomplete_seeds:
        incomplete_text = ', '.join(str(int(s)) for s in incomplete_seeds)
        tex.append(rf'Only seeds {complete_seed_text} completed all five epochs; seed {incomplete_text} stopped early and is excluded from this final-epoch summary.')
    else:
        tex.append(rf'The table reports final-epoch diagnostics over completed seeds {complete_seed_text}.')
    tex.append(r'}')
    tex.append(r'\label{tab:adaptive_budget_ablation}')
    tex.append(r'\end{table}')
    with open(os.path.join(logs_dir, 'adaptive_budget_ablation_table.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    # Plots.
    plt.rcParams.update({'figure.figsize': (5.4, 3.6), 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25})
    plot_specs = [
        ('test_acc', 'test accuracy', 'test_accuracy_by_policy.png'),
        ('avg_steps', 'average Sinkhorn steps/layer', 'avg_steps_by_policy.png'),
        ('epoch_time_sec', 'time per epoch (s)', 'time_per_epoch_by_policy.png'),
        ('cert_bound', r'product certificate $\prod_\ell(1+\varepsilon_c^{(\ell)})$', 'product_cert_by_policy.png'),
        ('exact_T1', r'exact $\|T_L\|_1$', 'exact_T1_by_policy.png'),
        ('cap_failure_rate', 'cap-failure rate', 'cap_failure_by_policy.png'),
    ]
    for field, ylabel, fname in plot_specs:
        fig, ax = plt.subplots()
        for policy in POLICIES:
            sub = df[df['policy'] == policy]
            grouped = sub.groupby('epoch')[field].agg(['mean', 'std']).reset_index()
            x = grouped['epoch'].to_numpy(dtype=float) + 1
            y = grouped['mean'].to_numpy(dtype=float)
            sd = grouped['std'].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, marker='o', label=POLICY_LABELS[policy].replace('$', ''))
            if len(x) > 1:
                ax.fill_between(x, y - sd, y + sd, alpha=0.12)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=220, bbox_inches='tight')
        plt.close(fig)

    with open(os.path.join(logs_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    zip_root = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.dirname(out_dir)
    zip_path = os.path.join(zip_root, 'adaptive-budget-ablation.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                path = os.path.join(root, name)
                zf.write(path, os.path.relpath(path, os.path.dirname(out_dir)))
    print(f'\n[outputs] {out_dir}')
    print(f'[zip] {zip_path}')
    print('\nFinal summary:')
    display(summary)


def run_ablation(cfg: Dict[str, object]) -> None:
    cfg = dict(cfg)
    out_dir = str(cfg['out_dir'])
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f'[device] {device}, n_gpus={n_gpus}')
    print(f'[policies] {POLICIES}')
    print(f"[budget] B1=(1+eps)^L={(1.0 + float(cfg['sinkhorn_eps_tol'])) ** int(cfg['depth']):.6f}")

    all_rows: List[Dict[str, object]] = []
    for seed in list(cfg['seeds']):
        print(f'\n=== seed {seed} ===')
        set_global_seed(int(seed))
        train_loader, test_loader, sampler = get_loaders(cfg, int(seed))
        models = make_policy_models(cfg, device)
        if bool(cfg.get('data_parallel', False)) and n_gpus >= 2:
            models = {p: nn.DataParallel(m) for p, m in models.items()}
        optimizers = {
            p: torch.optim.SGD(m.parameters(), lr=float(cfg['lr']), momentum=float(cfg['momentum']), weight_decay=float(cfg['weight_decay']))
            for p, m in models.items()
        }
        scalers = {
            p: torch.cuda.amp.GradScaler(enabled=(bool(cfg['amp']) and device.type == 'cuda'))
            for p in POLICIES
        }

        for epoch in range(int(cfg['epochs'])):
            sampler.set_epoch(epoch)
            train_stats = train_one_epoch(models, optimizers, scalers, train_loader, device, cfg)
            for policy in POLICIES:
                test_loss, test_acc, eval_time = evaluate(models[policy], test_loader, device, cfg)
                stats = collect_policy_stats(models[policy])
                train_loss, train_acc, train_time = train_stats[policy]
                row = {
                    'seed': int(seed),
                    'epoch': int(epoch),
                    'policy': policy,
                    'train_loss': float(train_loss),
                    'train_acc': float(train_acc),
                    'test_loss': float(test_loss),
                    'test_acc': float(test_acc),
                    'train_time_sec': float(train_time),
                    'eval_time_sec': float(eval_time),
                    'epoch_time_sec': float(train_time + eval_time),
                    **{k: float(v) for k, v in stats.items()},
                }
                all_rows.append(row)
                print(
                    f"epoch {epoch:02d} {policy:8s} "
                    f"acc={test_acc:.3f} avg_steps={stats['avg_steps']:.2f} "
                    f"cert={stats['cert_bound']:.4g} exactT1={stats['exact_T1']:.4g} "
                    f"cap_fail={stats['cap_failure_rate']:.3g} time={train_time + eval_time:.1f}s"
                )
            write_summary_outputs(all_rows, cfg)

    write_summary_outputs(all_rows, cfg)


if __name__ == '__main__':
    # Start the run. Edit CONFIG above first if you want a quick one-seed run.
    run_ablation(CONFIG)
