import argparse
import glob
import os
import sys
import subprocess
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.visualization.plot_vis_utils import (
    colors_for_instance_labels,
    has_visible_rgb,
    maybe_use_reference_coords,
    parse_instance_metrics_from_eval,
    parse_instance_metrics_for_scene,
    read_xy_labels_from_ply,
    read_xy_rgb_from_las,
    read_xy_rgb_from_ply,
    resolve_scene_las,
    resolve_scene_ply,
    sample_idx,
)


def draw_blank(ax, title, message):
    ax.set_title(title, fontsize=9)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f4f4f4")


def get_default_scenes(gt_dir):
    files = sorted(glob.glob(os.path.join(gt_dir, "test_*.ply")))
    scenes = [os.path.splitext(os.path.basename(p))[0] for p in files]
    return scenes[:6]


def maybe_run_eval(eval_dir):
    if not eval_dir or not os.path.isdir(eval_dir):
        return
    ply_files = sorted(glob.glob(os.path.join(eval_dir, "*.ply")))
    if not ply_files:
        return
    scene_names = [os.path.splitext(os.path.basename(p))[0] for p in ply_files]
    missing = [
        s for s in scene_names if not os.path.exists(os.path.join(eval_dir, f"{s}_evaluation_test.txt"))
    ]
    if not missing:
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cmd = [sys.executable, os.path.join(repo_root, "tools", "evaluation", "final_eval.py"), eval_dir]
    subprocess.run(cmd, cwd=repo_root, check=True)


def fmt_metrics(metrics):
    if not metrics:
        return "inst eval: n/a"
    return (
        f"F1 {metrics.get('F1', float('nan')):.2f} | "
        f"P {metrics.get('mPrecision', float('nan')):.2f} | "
        f"R {metrics.get('mRecall', float('nan')):.2f} | "
        f"PQt {metrics.get('meanPQ_things', float('nan')):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Create a 6x6 test matrix: rows=scenes, cols=views/models.")
    parser.add_argument("--gt-dir", default="data/labeled/test", help="Directory with GT PLY files.")
    parser.add_argument("--watershed-dir", default="work_dirs/watershed", help="Watershed output directory.")
    parser.add_argument("--pretrained-dir", default="work_dirs/pretrained", help="Pretrained output directory.")
    parser.add_argument("--xyz-dir", default="work_dirs/xyz", help="XYZ output directory.")
    parser.add_argument("--xyz-rgb-dir", default="work_dirs/xyzrgb", help="XYZ+RGB output directory.")
    parser.add_argument(
        "--rgb-las-dirs",
        default="data/raw/test,data/intermediate",
        help="Comma-separated LAS roots used as RGB fallback if GT PLY RGB is empty.",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene stems (example: test_52,test_54,...). Defaults to first 6 in gt-dir.",
    )
    parser.add_argument(
        "--out-file",
        default="work_dirs/plots/test_matrix_6x6.png",
        help="Output PNG path.",
    )
    parser.add_argument("--max-points", type=int, default=160000, help="Max sampled points per panel.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--point-size", type=float, default=0.75, help="Scatter point size.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    parser.add_argument(
        "--ensure-eval",
        action="store_true",
        help="Run final_eval on model dirs if scene eval files are missing.",
    )
    args = parser.parse_args()

    gt_dir = os.path.abspath(args.gt_dir)
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = get_default_scenes(gt_dir)

    # Force 6 rows. Missing rows become blank.
    scenes = scenes[:6]
    while len(scenes) < 6:
        scenes.append(f"missing_{len(scenes)+1}")

    cols = [
        ("RGB Overhead", "rgb", gt_dir),
        ("GT", "inst", gt_dir),
        ("Watershed", "inst", args.watershed_dir),
        ("Pretrained", "inst", args.pretrained_dir),
        ("XYZ", "inst", args.xyz_dir),
        ("XYZ_RGB", "inst", args.xyz_rgb_dir),
    ]

    fig, axes = plt.subplots(6, 6, figsize=(19, 19), constrained_layout=True)
    rgb_las_dirs = [p.strip() for p in args.rgb_las_dirs.split(",") if p.strip()]

    # Build panel file map first so we can infer eval dirs and overall metrics per model column.
    panel_ply = [[None for _ in range(6)] for _ in range(6)]
    for r, scene in enumerate(scenes):
        for c, (_, _, src_dir) in enumerate(cols):
            panel_ply[r][c] = resolve_scene_ply(src_dir, scene)

    # Optional eval refresh for model columns.
    if args.ensure_eval:
        for c in [2, 3, 4, 5]:
            dirs = sorted({os.path.dirname(p) for p in [panel_ply[r][c] for r in range(6)] if p})
            for d in dirs:
                maybe_run_eval(d)

    # Load per-scene metrics for each model column and compute column-wise overall means.
    scene_metrics = defaultdict(dict)  # scene_metrics[col][scene] = dict
    col_overall = {}
    for c in [2, 3, 4, 5]:
        bucket = []
        for r, scene in enumerate(scenes):
            p = panel_ply[r][c]
            if not p:
                continue
            eval_txt = os.path.join(os.path.dirname(p), f"{scene}_evaluation_test.txt")
            m = parse_instance_metrics_from_eval(eval_txt)
            if not m:
                m = parse_instance_metrics_for_scene(os.path.dirname(p), scene)
            if m:
                scene_metrics[c][scene] = m
                bucket.append(m)
        if bucket:
            col_overall[c] = {
                "F1": sum(b.get("F1", 0.0) for b in bucket) / len(bucket),
                "mPrecision": sum(b.get("mPrecision", 0.0) for b in bucket) / len(bucket),
                "mRecall": sum(b.get("mRecall", 0.0) for b in bucket) / len(bucket),
                "meanPQ_things": sum(b.get("meanPQ_things", 0.0) for b in bucket) / len(bucket),
            }
        else:
            col_overall[c] = None

    for r, scene in enumerate(scenes):
        gt_ref_x = None
        gt_ref_y = None
        gt_ref_path = resolve_scene_ply(gt_dir, scene)
        if gt_ref_path:
            try:
                gt_ref_x, gt_ref_y, _, _ = read_xy_labels_from_ply(gt_ref_path, "treeID")
            except Exception:
                gt_ref_x, gt_ref_y = None, None

        for c, (col_name, mode, src_dir) in enumerate(cols):
            ax = axes[r, c]
            if r == 0:
                extra = ""
                if c in col_overall:
                    extra = "\n" + fmt_metrics(col_overall[c])
                ax.set_title(f"{col_name}{extra}", fontsize=10)

            try:
                if mode == "rgb":
                    ply_path = resolve_scene_ply(src_dir, scene)
                    if not ply_path:
                        raise FileNotFoundError("No PLY")
                    x, y, rgb = read_xy_rgb_from_ply(ply_path)
                    if not has_visible_rgb(rgb):
                        las_path = resolve_scene_las(rgb_las_dirs, scene)
                        if las_path:
                            lx, ly, lrgb = read_xy_rgb_from_las(las_path)
                            x, y, rgb = lx, ly, lrgb
                    idx = sample_idx(x.shape[0], args.max_points, args.seed + r * 23 + c * 31)
                    ax.scatter(
                        x[idx],
                        y[idx],
                        c=rgb[idx],
                        s=args.point_size,
                        alpha=0.95,
                        linewidths=0,
                        rasterized=True,
                    )
                else:
                    ply_path = resolve_scene_ply(src_dir, scene)
                    if not ply_path:
                        raise FileNotFoundError("No PLY")
                    x, y, labels, _ = read_xy_labels_from_ply(ply_path, None)
                    if gt_ref_x is not None and gt_ref_y is not None:
                        x, y, _ = maybe_use_reference_coords(x, y, gt_ref_x, gt_ref_y)
                    idx = sample_idx(x.shape[0], args.max_points, args.seed + r * 23 + c * 31)
                    ax.scatter(
                        x[idx],
                        y[idx],
                        c=colors_for_instance_labels(labels[idx]),
                        s=args.point_size,
                        alpha=0.92,
                        linewidths=0,
                        rasterized=True,
                    )
            except Exception:
                draw_blank(ax, col_name if r != 0 else ax.get_title(), "Load error")
                continue

            if c == 0:
                ax.set_ylabel(f"Sample {r+1}\n{scene}", fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

            # Add per-scene instance metrics under model panels.
            if c in [2, 3, 4, 5]:
                m = scene_metrics.get(c, {}).get(scene)
                ax.text(
                    0.5,
                    -0.10,
                    fmt_metrics(m),
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7,
                    clip_on=False,
                    color="#222222",
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    fig.savefig(args.out_file, dpi=args.dpi)
    plt.close(fig)
    print(f"Done. Wrote {args.out_file}")


if __name__ == "__main__":
    main()
