import argparse
import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def sample_idx(n_points, max_points, seed):
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def find_label_field(names, preferred, candidates):
    if preferred and preferred in names:
        return preferred
    for name in candidates:
        if name in names:
            return name
    return None


def stem_candidates(pred_stem):
    cands = [pred_stem]
    cands.append(re.sub(r"_final_results$", "", pred_stem))
    cands.append(re.sub(r"_round\d+$", "", pred_stem))
    cands.append(re.sub(r"_bluepoints_\d+$", "", pred_stem))
    cands.append(re.sub(r"_\d+$", "", pred_stem))
    out = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def resolve_gt_file(pred_file, gt_dir):
    pred_stem = os.path.splitext(os.path.basename(pred_file))[0]
    for cand in stem_candidates(pred_stem):
        path = os.path.join(gt_dir, f"{cand}.ply")
        if os.path.exists(path):
            return path
    return None


def read_xy_labels(ply_path, label_field):
    ply = PlyData.read(ply_path)
    data = ply["vertex"].data
    names = data.dtype.names
    label = find_label_field(names, label_field, ["instance_pred", "instance_gt", "treeID"])
    if label is None:
        raise ValueError(f"No supported label field found in {ply_path}: {names}")
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    labels = np.asarray(data[label], dtype=np.int64)
    return x, y, labels, label


def _is_half_meter_quantized(values, sample_size=50000):
    if values.size == 0:
        return False
    step = max(1, values.size // sample_size)
    sample = np.asarray(values[::step], dtype=np.float64)
    frac = np.mod(sample, 0.5)
    near_grid = np.isclose(frac, 0.0, atol=1e-6) | np.isclose(frac, 0.5, atol=1e-6)
    return np.mean(near_grid) > 0.999


def _maybe_use_pred_coords_for_gt(gt_x, gt_y, pred_x, pred_y):
    if gt_x.shape != pred_x.shape:
        return gt_x, gt_y, False
    if not (_is_half_meter_quantized(gt_x) or _is_half_meter_quantized(gt_y)):
        return gt_x, gt_y, False

    # Sanity check: if point ordering is aligned, GT and pred coordinates should
    # differ by at most quantization error (~0.25m for 0.5m grid).
    n = gt_x.shape[0]
    step = max(1, n // 50000)
    sx = np.median(np.abs(np.asarray(gt_x[::step], dtype=np.float64) - np.asarray(pred_x[::step], dtype=np.float64)))
    sy = np.median(np.abs(np.asarray(gt_y[::step], dtype=np.float64) - np.asarray(pred_y[::step], dtype=np.float64)))
    if sx <= 0.26 and sy <= 0.26:
        return pred_x, pred_y, True
    return gt_x, gt_y, False


def collect_pairs(pred_files, gt_dir):
    pairs = []
    skipped = 0
    for pred_file in pred_files:
        gt_file = resolve_gt_file(pred_file, gt_dir)
        if gt_file is None:
            print(f"[skip] No GT match for {os.path.basename(pred_file)}")
            skipped += 1
            continue
        pairs.append((pred_file, gt_file))
    return pairs, skipped


def plot_all_pairs(pairs, out_file, max_points, seed, pred_field, gt_field, point_size, dpi):
    n = len(pairs)
    fig_h = max(4, int(math.ceil(n * 2.8)))
    fig, axes = plt.subplots(n, 2, figsize=(14, fig_h), squeeze=False, constrained_layout=True)
    last_scatter = None

    for i, (pred_file, gt_file) in enumerate(pairs):
        pred_x, pred_y, pred_labels, pred_used = read_xy_labels(pred_file, pred_field)
        gt_x, gt_y, gt_labels, gt_used = read_xy_labels(gt_file, gt_field)
        gt_plot_x, gt_plot_y, used_pred_coords_for_gt = _maybe_use_pred_coords_for_gt(
            gt_x, gt_y, pred_x, pred_y
        )

        p_idx = sample_idx(pred_x.shape[0], max_points, seed + i * 17)
        g_idx = sample_idx(gt_x.shape[0], max_points, seed + i * 17 + 1)

        ax_gt = axes[i, 0]
        ax_pred = axes[i, 1]

        last_scatter = ax_gt.scatter(
            gt_plot_x[g_idx],
            gt_plot_y[g_idx],
            c=np.mod(gt_labels[g_idx], 256),
            s=point_size,
            cmap="tab20",
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        gt_title_suffix = " + pred_coords" if used_pred_coords_for_gt else ""
        ax_gt.set_title(f"GT: {os.path.basename(gt_file)} [{gt_used}{gt_title_suffix}]")
        ax_gt.set_xlabel("X")
        ax_gt.set_ylabel("Y")
        ax_gt.set_aspect("equal", adjustable="box")

        ax_pred.scatter(
            pred_x[p_idx],
            pred_y[p_idx],
            c=np.mod(pred_labels[p_idx], 256),
            s=point_size,
            cmap="tab20",
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )
        ax_pred.set_title(f"Pred: {os.path.basename(pred_file)} [{pred_used}]")
        ax_pred.set_xlabel("X")
        ax_pred.set_ylabel("Y")
        ax_pred.set_aspect("equal", adjustable="box")

    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes.ravel().tolist(),
            fraction=0.01,
            pad=0.01,
            label="instance id (mod 256)",
        )

    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot all prediction .ply files next to matching ground-truth .ply files in one subplot figure."
    )
    parser.add_argument("--work-dir", required=True, help="Directory containing predicted .ply files.")
    parser.add_argument(
        "--gt-dir",
        default="data/labeled/test",
        help="Directory containing ground-truth .ply files (default: data/labeled/test).",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Output PNG path (default: <work-dir>/plots/gt_vs_pred/all_gt_vs_pred.png).",
    )
    parser.add_argument("--max-points", type=int, default=200000, help="Max sampled points per cloud.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--pred-field", default=None, help="Prediction label field (auto if omitted).")
    parser.add_argument("--gt-field", default=None, help="Ground truth label field (auto if omitted).")
    parser.add_argument("--point-size", type=float, default=0.35, help="Scatter point size.")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    gt_dir = os.path.abspath(args.gt_dir)

    out_file = args.out_file or os.path.join(work_dir, "plots", "gt_vs_pred", "all_gt_vs_pred.png")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(work_dir, "*.ply")))
    if not pred_files:
        raise FileNotFoundError(f"No prediction .ply files found in {work_dir}")

    pairs, skipped = collect_pairs(pred_files, gt_dir)
    if not pairs:
        raise FileNotFoundError(f"No matched pred/gt pairs. work_dir={work_dir}, gt_dir={gt_dir}")

    plot_all_pairs(
        pairs=pairs,
        out_file=out_file,
        max_points=args.max_points,
        seed=args.seed,
        pred_field=args.pred_field,
        gt_field=args.gt_field,
        point_size=args.point_size,
        dpi=args.dpi,
    )
    print(f"Done. matched={len(pairs)}, skipped={skipped}, out_file={out_file}")


if __name__ == "__main__":
    main()
