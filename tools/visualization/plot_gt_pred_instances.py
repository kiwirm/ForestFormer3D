import argparse
import os
import re

import numpy as np
from plyfile import PlyData

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

GOLDEN_RATIO = 0.61803398875


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GT vs Pred instance segmentation top-down per tile."
    )
    parser.add_argument(
        "--pred-dir",
        default="work_dirs/cass_pretrained_infer",
        help="Directory containing predicted .ply files with instance_gt/instance_pred.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/plots/instance_gt_pred",
        help="Directory to write PNGs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of tiles to plot.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles that already have an output PNG.",
    )
    return parser.parse_args()


def read_tile(ply_path):
    ply = PlyData.read(ply_path)
    data = ply["vertex"].data
    names = data.dtype.names
    required = {"x", "y", "instance_gt", "instance_pred"}
    if not required.issubset(set(names)):
        missing = ", ".join(sorted(required - set(names)))
        raise ValueError(f"Missing required fields in {ply_path}: {missing}")
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    inst_gt = np.asarray(data["instance_gt"], dtype=np.int64)
    inst_pred = np.asarray(data["instance_pred"], dtype=np.int64)
    return x, y, inst_gt, inst_pred


def distinct_colors(ids):
    ids = np.asarray(ids, dtype=np.int64)
    hues = (ids * GOLDEN_RATIO) % 1.0
    s = np.full_like(hues, 0.85, dtype=np.float32)
    v = np.full_like(hues, 0.95, dtype=np.float32)
    hsv = np.stack([hues, s, v], axis=1)
    return hsv_to_rgb(hsv)


def colors_for_labels(labels):
    unique_ids = np.unique(labels)
    colors = distinct_colors(unique_ids)
    idx = np.searchsorted(unique_ids, labels)
    point_colors = colors[idx]
    bg_mask = labels <= 0
    if np.any(bg_mask):
        point_colors[bg_mask] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return point_colors


def parse_eval_stats(eval_path):
    if not os.path.exists(eval_path):
        return None, None
    rq = None
    sq = None
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Instance Segmentation meanRQ:"):
                try:
                    rq = float(line.strip().split(":", 1)[1])
                except ValueError:
                    rq = None
            elif line.startswith("Instance Segmentation meanSQ:"):
                try:
                    sq = float(line.strip().split(":", 1)[1])
                except ValueError:
                    sq = None
    return rq, sq


def plot_tile(tile_name, ply_path, eval_path, out_path, point_size, dpi):
    x, y, inst_gt, inst_pred = read_tile(ply_path)

    gt_colors = colors_for_labels(inst_gt)
    pred_colors = colors_for_labels(inst_pred)

    rq, sq = parse_eval_stats(eval_path)
    gt_instances = int(np.sum(np.unique(inst_gt) > 0))
    pred_instances = int(np.sum(np.unique(inst_pred) > 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)

    axes[0].scatter(x, y, s=point_size, c=gt_colors, linewidths=0, rasterized=True)
    axes[0].set_title("GT instances")
    axes[1].scatter(x, y, s=point_size, c=pred_colors, linewidths=0, rasterized=True)
    axes[1].set_title("Pred instances")

    for ax in axes:
        ax.set_aspect("equal", "box")
        ax.axis("off")

    rq_text = f"RQ: {rq:.4f}" if rq is not None else "RQ: n/a"
    sq_text = f"SQ: {sq:.4f}" if sq is not None else "SQ: n/a"
    fig.suptitle(tile_name)
    fig.text(
        0.5,
        0.02,
        f"{rq_text}  |  {sq_text}  |  GT inst: {gt_instances}  |  Pred inst: {pred_instances}",
        ha="center",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    pred_dir = args.pred_dir
    out_dir = args.out_dir

    if not os.path.isdir(pred_dir):
        raise SystemExit(f"pred-dir not found: {pred_dir}")

    ply_files = [
        f
        for f in os.listdir(pred_dir)
        if f.endswith(".ply") and not f.endswith("_gt.ply")
    ]
    ply_files.sort()

    if args.limit is not None:
        ply_files = ply_files[: args.limit]

    if not ply_files:
        raise SystemExit(f"No .ply files found in {pred_dir}")

    for fname in ply_files:
        tile_name = os.path.splitext(fname)[0]
        ply_path = os.path.join(pred_dir, fname)
        eval_path = os.path.join(pred_dir, f"{tile_name}_evaluation_test.txt")
        out_path = os.path.join(out_dir, f"{tile_name}_gt_pred.png")
        if args.skip_existing and os.path.exists(out_path):
            continue
        plot_tile(tile_name, ply_path, eval_path, out_path, args.point_size, args.dpi)


if __name__ == "__main__":
    main()
