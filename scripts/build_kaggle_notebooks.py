#!/usr/bin/env python3
"""Build self-contained Kaggle notebooks from the release scripts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def build(name, modules, command):
    intro = (
        "# Finite-Step Sinkhorn Certificates\n\n"
        "Enable a GPU and attach the [CIFAR-10 Python dataset]"
        "(https://www.kaggle.com/datasets/pankrzysiu/cifar10-python). "
        "Run the cells in order. Helper files are written into the working directory.\n\n"
        "ChatGPT and OpenAI Codex assisted with code development and debugging, "
        "experiment and figure scripts, and manuscript editing. The author takes "
        "responsibility for the paper and the accompanying code.\n"
    )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}]
    for module in modules:
        source = (ROOT / "code" / module).read_text()
        cells.append(code_cell(f"%%writefile {module}\n{source}"))
    cells.append(code_cell(command + "\n"))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    target = ROOT / "notebooks" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1) + "\n")
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    helpers = ["cifar_data.py", "adaptive_budget_ablation_kaggle.py", "paper_extra_experiments_kaggle.py"]
    build("paper_extra_experiments_kaggle.ipynb", helpers,
          "%run paper_extra_experiments_kaggle.py")
    build("paper_extra_experiments_3seeds_5epochs_kaggle.ipynb",
          helpers + ["paper_extra_experiments_3seeds_5epochs_kaggle.py"],
          "%run paper_extra_experiments_3seeds_5epochs_kaggle.py")
    build("transformer_relevance_3seeds_8epochs.ipynb", ["tiny_transformer_relevance_demo.py"],
          "%run tiny_transformer_relevance_demo.py --data_dir /kaggle/input/cifar10-python "
          "--out_dir /kaggle/working/relevance-demo --seeds 0 1 2 --epochs 8 "
          "--train_size 5000 --test_size 2000 --depth 24 --n_streams 8 --dim 24 "
          "--heads 2 --batch_size 128 --sinkhorn_t_max 40 --sinkhorn_eps 5e-4 --save_mixers")
