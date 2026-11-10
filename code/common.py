"""Shared reproducibility, plotting, and synthetic-data helpers."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
COLORS = ["#167D8D", "#B05B35", "#17334D", "#9772A4", "#75A46A"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight",
                     "axes.prop_cycle": plt.cycler(color=COLORS),
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def setup(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--figures", type=Path, default=PROJECT / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    return args, np.random.default_rng(args.seed)


def versions():
    result = {"python": platform.python_version()}
    for package in ["numpy", "scipy", "scikit-learn", "matplotlib", "torch",
                    "ripser", "persim", "umap-learn", "numba"]:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def record(args, name: str, metrics: dict):
    payload = {"tutorial": name, "seed": args.seed, "versions": versions(),
               "metrics": metrics}
    path = args.out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tutorial": name, "metrics": metrics}, allow_nan=False))


def save_figure(fig, args, name):
    fig.savefig(args.figures / f"{name}.pdf")
    fig.savefig(args.figures / f"{name}.png", dpi=160)
    plt.close(fig)


def circle(rng, n=300, noise=0.03):
    theta = rng.uniform(0, 2 * np.pi, n)
    x = np.column_stack([np.cos(theta), np.sin(theta)])
    return x + rng.normal(0, noise, x.shape), theta


def sphere(rng, n=220, noise=0.015):
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x + rng.normal(0, noise, x.shape)


def torus(rng, n=350, noise=0.015):
    # Uniform angles are deliberately NOT uniform surface-area sampling.
    a, b = rng.uniform(0, 2 * np.pi, (2, n))
    x = np.column_stack([(2 + 0.7 * np.cos(b)) * np.cos(a),
                         (2 + 0.7 * np.cos(b)) * np.sin(a), 0.7 * np.sin(b)])
    return x + rng.normal(0, noise, x.shape)


def longest_bar(diagram):
    finite = diagram[np.isfinite(diagram[:, 1])]
    return float(np.max(finite[:, 1] - finite[:, 0], initial=0.0))
