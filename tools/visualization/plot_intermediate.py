import argparse
import glob
import math
import os

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_idx(n_points, max_points, seed):
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def _labels_from_las(las, idx):
    if "tree_id" in las.point_format.extra_dimension_names:
        return np.asarray(las["tree_id"])[idx]
    return np.zeros(len(idx), dtype=np.int64)


def collect_files(mode):
    if mode == "train":
        candidates = sorted(glob.glob("data/intermediate/train*.las"))
        return candidates
    return sorted(glob.glob("data/intermediate/test_*.las"))


def make_subplot_figure(files, output, max_points, seed, mode):
    n = len(files)
    cols = min(3, n) if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, constrained_layout=True
    )
    axes_flat = axes.ravel()
    last_scatter = None

    for i, f in enumerate(files):
        ax = axes_flat[i]
        las = laspy.read(f)
        n_points = len(las.x)
        idx = sample_idx(n_points, max_points, seed + i)
        x = np.asarray(las.x)[idx]
        y = np.asarray(las.y)[idx]
        labels = _labels_from_las(las, idx)

        last_scatter = ax.scatter(
            x,
            y,
            c=np.mod(labels, 256),
            s=0.35,
            cmap="tab20",
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(f"{os.path.basename(f)} ({len(idx):,}/{n_points:,})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    if last_scatter is not None:
        fig.colorbar(last_scatter, ax=axes_flat[:n].tolist(), fraction=0.02, pad=0.01, label="tree_id (mod 256)")
    fig.suptitle(f"Intermediate {mode} LAS overview", fontsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot intermediate LAS files as a single subplot figure.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Plot train intermediates.")
    mode_group.add_argument("--test", action="store_true", help="Plot test intermediates.")
    parser.add_argument("--max-points", type=int, default=200000, help="Max sampled points per tile.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output png path.")
    args = parser.parse_args()

    mode = "train" if args.train else "test"
    files = collect_files(mode)
    if not files:
        raise FileNotFoundError(f"No intermediate LAS files found for mode '{mode}'.")

    default_out = f"work_dirs/plots/intermediate_{mode}.png"
    output = args.output or default_out
    os.makedirs(os.path.dirname(output), exist_ok=True)

    make_subplot_figure(files, output, args.max_points, args.seed, mode)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
