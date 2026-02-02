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
        description="Plot GT vs prediction instance segmentation per tile for a run work_dir."
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/cass_pretrained_infer",
        help="Run directory containing output .ply files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write PNGs. Default: <work-dir>/plots/instance_gt_pred",
    )
    parser.add_argument(
        "--gt-field",
        default="instance_gt",
        help="Field name to use as GT labels.",
    )
    parser.add_argument(
        "--pred-field",
        default="instance_pred",
        help="Field name to use as prediction labels.",
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


def read_tile(ply_path, gt_field, pred_field):
    ply = PlyData.read(ply_path)
    data = ply["vertex"].data
    names = data.dtype.names
    required = {"x", "y", gt_field, pred_field}
    if not required.issubset(set(names)):
        missing = ", ".join(sorted(required - set(names)))
        raise ValueError(f"Missing required fields in {ply_path}: {missing}")
    # Keep double precision to avoid meter-scale quantization on large map coords (e.g. NZTM y~5e6).
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    inst_gt = np.asarray(data[gt_field], dtype=np.int64)
    inst_pred = np.asarray(data[pred_field], dtype=np.int64)
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


def plot_tile(tile_name, ply_path, eval_path, out_path, point_size, dpi, gt_field, pred_field):
    x, y, inst_gt, inst_pred = read_tile(ply_path, gt_field, pred_field)
    # Plot in local tile coordinates to avoid precision artifacts with large absolute coordinates.
    x = x - np.min(x)
    y = y - np.min(y)

    gt_colors = colors_for_labels(inst_gt)
    pred_colors = colors_for_labels(inst_pred)

    rq, sq = parse_eval_stats(eval_path)
    gt_instances = int(np.sum(np.unique(inst_gt) > 0))
    pred_instances = int(np.sum(np.unique(inst_pred) > 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)

    axes[0].scatter(x, y, s=point_size, c=gt_colors, linewidths=0, rasterized=True)
    axes[0].set_title(f"GT ({gt_field})")
    axes[1].scatter(x, y, s=point_size, c=pred_colors, linewidths=0, rasterized=True)
    axes[1].set_title(f"Pred ({pred_field})")

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
    work_dir = args.work_dir
    out_dir = args.out_dir or os.path.join(work_dir, "plots", "instance_gt_pred")

    if not os.path.isdir(work_dir):
        raise SystemExit(f"work-dir not found: {work_dir}")

    ply_files = [
        f
        for f in os.listdir(work_dir)
        if f.endswith(".ply") and not f.endswith("_gt.ply")
    ]
    ply_files.sort()

    if args.limit is not None:
        ply_files = ply_files[: args.limit]

    if not ply_files:
        raise SystemExit(f"No .ply files found in {work_dir}")

    for fname in ply_files:
        tile_name = os.path.splitext(fname)[0]
        ply_path = os.path.join(work_dir, fname)
        eval_path = os.path.join(work_dir, f"{tile_name}_evaluation_test.txt")
        out_path = os.path.join(out_dir, f"{tile_name}_gt_pred.png")
        if args.skip_existing and os.path.exists(out_path):
            continue
        plot_tile(
            tile_name,
            ply_path,
            eval_path,
            out_path,
            args.point_size,
            args.dpi,
            args.gt_field,
            args.pred_field,
        )


if __name__ == "__main__":
    main()
