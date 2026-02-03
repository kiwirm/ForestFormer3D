import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from mmengine.config import Config
from mmengine.runner import Runner


GOLDEN_RATIO = 0.61803398875


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-sample inference from a checkpoint and save GT vs pred instance plot."
    )
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint path")
    parser.add_argument("--work-dir", required=True, help="Runner work dir")
    parser.add_argument("--out-png", required=True, help="Output PNG path")
    parser.add_argument("--max-points-plot", type=int, default=1200000,
                        help="Randomly subsample this many points for plotting")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pred_masks_to_point_ids(mask_array, score_array):
    if mask_array.ndim == 1:
        mask_array = mask_array[None, :]
    num_pred, num_points = mask_array.shape
    if num_pred == 0:
        return np.full((num_points,), -1, dtype=np.int64)

    score_array = score_array.reshape(-1)
    order = np.argsort(score_array)  # low->high, high score overwrites last
    out = np.full((num_points,), -1, dtype=np.int64)
    for new_id, idx in enumerate(order):
        out[mask_array[idx].astype(bool)] = int(new_id)
    return out


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = args.work_dir
    cfg.test_dataloader.num_workers = 0

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.eval()

    batch = next(iter(runner.test_dataloader))
    with torch.no_grad():
        outputs = runner.model.test_step(batch, epoch=0)

    sample = outputs[0]
    points = to_numpy(batch["inputs"]["points"][0])[:, :2]
    gt_ids = to_numpy(sample.eval_ann_info["pts_instance_mask"]).astype(np.int64)

    pred_seg = sample.pred_pts_seg
    pred_masks = to_numpy(pred_seg.pts_instance_mask)
    pred_scores = to_numpy(pred_seg.instance_scores).astype(np.float32)
    pred_ids = pred_masks_to_point_ids(pred_masks, pred_scores)

    n_points = points.shape[0]
    if n_points > args.max_points_plot:
        rng = np.random.default_rng(args.seed)
        choice = rng.choice(n_points, size=args.max_points_plot, replace=False)
        points = points[choice]
        gt_ids = gt_ids[choice]
        pred_ids = pred_ids[choice]

    x = points[:, 0] - np.min(points[:, 0])
    y = points[:, 1] - np.min(points[:, 1])
    gt_colors = colors_for_labels(gt_ids)
    pred_colors = colors_for_labels(pred_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=220)
    axes[0].scatter(x, y, s=0.5, c=gt_colors, linewidths=0, rasterized=True)
    axes[1].scatter(x, y, s=0.5, c=pred_colors, linewidths=0, rasterized=True)
    axes[0].set_title("GT instances")
    axes[1].set_title("Pred instances")

    for ax in axes:
        ax.set_aspect("equal", "box")
        ax.axis("off")

    gt_inst = int(np.sum(np.unique(gt_ids) > 0))
    pred_inst = int(np.sum(np.unique(pred_ids) > 0))
    fig.suptitle(
        f"{os.path.basename(args.checkpoint)} | GT inst: {gt_inst} | Pred inst: {pred_inst}"
    )

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
