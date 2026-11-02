"""Locate the standard CIFAR-10 Python batches, including Kaggle mounts."""

from pathlib import Path

from torchvision.datasets import CIFAR10


class _DirectCIFAR10(CIFAR10):
    base_folder = ""


def load_cifar10(root: str, *, train: bool, transform):
    path = Path(root).expanduser()
    if (path / "cifar-10-batches-py/data_batch_1").is_file():
        return CIFAR10(str(path), train=train, download=False, transform=transform)
    candidates = [path] if (path / "data_batch_1").is_file() else []
    if path.is_dir() and not candidates:
        candidates = sorted({item.parent for item in path.rglob("data_batch_1")})
    if len(candidates) == 1:
        return _DirectCIFAR10(str(candidates[0]), train=train, download=False, transform=transform)
    if len(candidates) > 1:
        raise ValueError(f"Multiple CIFAR-10 directories found under {path}; select one explicitly.")
    return CIFAR10(str(path), train=train, download=True, transform=transform)
