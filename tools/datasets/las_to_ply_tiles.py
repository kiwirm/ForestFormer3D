import argparse
import os
from os import path as osp

import laspy
import numpy as np

from tools.support.plyutils import write_ply


def _get_rgb(las, n_points):
    if not hasattr(las, "red") or not hasattr(las, "green") or not hasattr(las, "blue"):
        return np.zeros((n_points, 3), dtype=np.float32)
    rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
    if rgb.max() > 255.0:
        rgb = rgb / 257.0
    return rgb


def _get_intensity(las, n_points, normalize):
    if not hasattr(las, "intensity"):
        return np.zeros((n_points, 1), dtype=np.float32)
    intensity = las.intensity.astype(np.float32).reshape(-1, 1)
    if normalize:
        max_val = np.max(intensity) if intensity.size > 0 else 0.0
        if max_val > 0:
            intensity = intensity / max_val
    return intensity


def _split_tiles(tile_ids, ratios, seed):
    rng = np.random.default_rng(seed)
    tile_ids = np.array(tile_ids, dtype=np.int64)
    rng.shuffle(tile_ids)
    n_tiles = tile_ids.shape[0]
    n_train = int(ratios[0] * n_tiles)
    n_val = int(ratios[1] * n_tiles)
    train_ids = set(tile_ids[:n_train].tolist())
    val_ids = set(tile_ids[n_train:n_train + n_val].tolist())
    test_ids = set(tile_ids[n_train + n_val:].tolist())
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser(
        description="Tile LAS/LAZ into PLYs and split into train/val/test."
    )
    parser.add_argument("input_las", help="Path to input LAS/LAZ file")
    parser.add_argument("--tile-size", type=float, default=30.0, help="Tile size in meters")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-points", type=int, default=1000, help="Skip tiles with fewer points")
    parser.add_argument("--semantic", type=int, default=0)
    parser.add_argument("--tree-id-dim", default="tree_id")
    parser.add_argument("--normalize-intensity", action="store_true")
    parser.add_argument("--split-prefix", default="cass")
    parser.add_argument("--train-dir", default="data/labeled/plys/train_val/cass")
    parser.add_argument("--test-dir", default="data/labeled/plys/test/cass")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()

    if not osp.exists(args.input_las):
        raise FileNotFoundError(f"Input LAS not found: {args.input_las}")

    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratios_sum, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    las = laspy.read(args.input_las)
    if args.tree_id_dim not in las.point_format.extra_dimension_names:
        raise ValueError(
            f"'{args.tree_id_dim}' not found in LAS extra dimensions. "
            f"Available: {las.point_format.extra_dimension_names}"
        )

    points = np.vstack((las.x, las.y, las.z)).astype(np.float32).T
    n_points = points.shape[0]
    tree_ids = np.asarray(getattr(las, args.tree_id_dim), dtype=np.int32)
    semantic_seg = np.full(tree_ids.shape, args.semantic, dtype=np.int32)

    rgb = _get_rgb(las, n_points)
    intensity = _get_intensity(las, n_points, args.normalize_intensity)

    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    ix = np.floor((points[:, 0] - xmin) / args.tile_size).astype(np.int64)
    iy = np.floor((points[:, 1] - ymin) / args.tile_size).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    tile_id = ix + iy * nx

    order = np.argsort(tile_id)
    tile_sorted = tile_id[order]

    # Find tile boundaries in sorted order
    boundaries = np.nonzero(np.diff(tile_sorted))[0] + 1
    splits = np.split(order, boundaries)
    tile_ids = []
    tile_indices = {}

    for inds in splits:
        tid = tile_id[inds[0]]
        if inds.size < args.min_points:
            continue
        tile_ids.append(tid)
        tile_indices[tid] = inds

    train_ids, val_ids, test_ids = _split_tiles(
        tile_ids,
        (args.train_ratio, args.val_ratio, args.test_ratio),
        args.seed,
    )

    if args.splits_dir == "data/splits" and args.split_prefix:
        args.splits_dir = osp.join(args.splits_dir, args.split_prefix)

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)

    xw = len(str(max(nx - 1, 0)))
    yw = len(str(max(ny - 1, 0)))

    train_list = []
    val_list = []
    test_list = []

    for tid in tile_ids:
        inds = tile_indices[tid]
        tile_ix = int(tid % nx)
        tile_iy = int(tid // nx)
        base = f"{args.split_prefix}_tile_x{tile_ix:0{xw}d}_y{tile_iy:0{yw}d}"

        if tid in train_ids:
            out_dir = args.train_dir
            train_list.append(base)
        elif tid in val_ids:
            out_dir = args.train_dir
            val_list.append(base)
        else:
            out_dir = args.test_dir
            test_list.append(base)

        fields = [points[inds]]
        field_names = ["x", "y", "z"]

        fields.append(rgb[inds])
        field_names += ["red", "green", "blue"]
        fields.append(intensity[inds])
        field_names += ["intensity"]

        fields += [semantic_seg[inds], tree_ids[inds]]
        field_names += ["semantic_seg", "treeID"]

        out_path = osp.join(out_dir, base + ".ply")
        ok = write_ply(out_path, fields, field_names)
        if not ok:
            raise RuntimeError(f"Failed to write PLY: {out_path}")

    prefix = args.split_prefix
    with open(osp.join(args.splits_dir, f"{prefix}_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_list) + "\n")
    with open(osp.join(args.splits_dir, f"{prefix}_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_list) + "\n")
    with open(osp.join(args.splits_dir, f"{prefix}_test_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_list) + "\n")

    print(
        f"Wrote {len(train_list)} train, {len(val_list)} val, {len(test_list)} test tiles. "
        f"Tile size: {args.tile_size}m"
    )


if __name__ == "__main__":
    main()
