import glob
import os
import re

import laspy
import matplotlib.colors as mcolors
import numpy as np
from plyfile import PlyData


def sample_idx(n_points, max_points, seed):
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def find_field(names, preferred=None, candidates=None):
    if preferred and preferred in names:
        return preferred
    for name in candidates or []:
        if name in names:
            return name
    return None


def resolve_scene_ply(root_dir, scene_name):
    if not root_dir:
        return None
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return None
    direct = os.path.join(root_dir, f"{scene_name}.ply")
    if os.path.exists(direct):
        return direct
    rec = sorted(glob.glob(os.path.join(root_dir, "**", f"{scene_name}.ply"), recursive=True))
    return rec[0] if rec else None


def read_xy_labels_from_ply(ply_path, label_field=None):
    data = PlyData.read(ply_path)["vertex"].data
    names = data.dtype.names
    used = find_field(names, label_field, ["instance_pred", "instance_gt", "treeID"])
    if used is None:
        raise ValueError(f"No supported label field in {ply_path}: {names}")
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    labels = np.asarray(data[used], dtype=np.int64)
    return x, y, labels, used


def read_xy_rgb_from_ply(ply_path):
    data = PlyData.read(ply_path)["vertex"].data
    names = data.dtype.names
    req = {"x", "y", "red", "green", "blue"}
    if not req.issubset(names):
        raise ValueError(f"Missing RGB fields in {ply_path}: {names}")
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    rgb = np.column_stack(
        [
            np.asarray(data["red"], dtype=np.float32),
            np.asarray(data["green"], dtype=np.float32),
            np.asarray(data["blue"], dtype=np.float32),
        ]
    )
    if np.nanmax(rgb) > 255.0:
        rgb = rgb / 257.0
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    return x, y, rgb


def has_visible_rgb(rgb):
    return np.any(rgb > 1e-6)


def read_xy_rgb_from_las(las_path):
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    if not all(hasattr(las, c) for c in ("red", "green", "blue")):
        raise ValueError(f"No RGB dimensions in {las_path}")
    rgb = np.column_stack(
        [
            np.asarray(las.red, dtype=np.float32),
            np.asarray(las.green, dtype=np.float32),
            np.asarray(las.blue, dtype=np.float32),
        ]
    )
    if np.nanmax(rgb) > 255.0:
        rgb = rgb / 257.0
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    return x, y, rgb


def resolve_scene_las(root_dirs, scene_name):
    for root_dir in root_dirs:
        if not root_dir:
            continue
        root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(root_dir):
            continue
        direct = os.path.join(root_dir, f"{scene_name}.las")
        if os.path.exists(direct):
            return direct
        rec = sorted(glob.glob(os.path.join(root_dir, "**", f"{scene_name}.las"), recursive=True))
        if rec:
            return rec[0]
    return None


def _min_positive_step(values, sample_size=80000):
    if values.size < 2:
        return np.inf
    step = max(1, values.size // sample_size)
    sample = np.asarray(values[::step], dtype=np.float64)
    uniq = np.unique(sample)
    if uniq.size < 2:
        return np.inf
    dif = np.diff(uniq)
    pos = dif[dif > 1e-8]
    if pos.size == 0:
        return np.inf
    return float(np.quantile(pos, 0.05))


def is_coarse_quantized(values, step_threshold=0.3):
    return _min_positive_step(values) >= step_threshold


def maybe_use_reference_coords(x, y, ref_x, ref_y):
    if x.shape != ref_x.shape:
        return x, y, False
    if not (is_coarse_quantized(x) or is_coarse_quantized(y)):
        return x, y, False
    n = x.shape[0]
    step = max(1, n // 50000)
    sx = x[::step]
    sy = y[::step]
    rx = ref_x[::step]
    ry = ref_y[::step]

    # Case 1: already in same frame.
    med_dx = np.median(np.abs(sx - rx))
    med_dy = np.median(np.abs(sy - ry))
    if med_dx <= 0.26 and med_dy <= 0.26:
        return ref_x, ref_y, True

    # Case 2: model uses a translated local frame (common for normalized coords).
    shift_x = np.median(sx - rx)
    shift_y = np.median(sy - ry)
    med_dx_shifted = np.median(np.abs((sx - shift_x) - rx))
    med_dy_shifted = np.median(np.abs((sy - shift_y) - ry))
    if med_dx_shifted <= 0.26 and med_dy_shifted <= 0.26:
        return ref_x, ref_y, True
    return x, y, False


def colors_for_instance_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.empty((labels.shape[0], 3), dtype=np.float32)
    bg_mask = labels < 0
    colors[bg_mask] = np.array([0.72, 0.72, 0.72], dtype=np.float32)

    fg_labels = np.unique(labels[~bg_mask])
    if fg_labels.size == 0:
        return colors
    h = (np.arange(fg_labels.size, dtype=np.float32) * 0.61803398875) % 1.0
    s = np.full_like(h, 0.72)
    v = np.full_like(h, 0.95)
    palette = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=1))
    idx_map = {int(lbl): i for i, lbl in enumerate(fg_labels.tolist())}
    mapped = np.fromiter((idx_map[int(vv)] for vv in labels[~bg_mask]), dtype=np.int64, count=np.sum(~bg_mask))
    colors[~bg_mask] = palette[mapped]
    return colors


def parse_instance_metrics_from_eval(eval_txt_path):
    if not eval_txt_path or not os.path.exists(eval_txt_path):
        return None
    wanted = {
        "Instance Segmentation mPrecision:": "mPrecision",
        "Instance Segmentation mRecall:": "mRecall",
        "Instance Segmentation F1 score:": "F1",
        "Instance Segmentation meanPQ (things):": "meanPQ_things",
        "Instance Segmentation mMUCov:": "mMUCov",
        "Instance Segmentation mMWCov:": "mMWCov",
    }
    out = {}
    with open(eval_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            for prefix, key in wanted.items():
                if s.startswith(prefix):
                    m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", s)
                    if m:
                        out[key] = float(m.group(1))
    return out if out else None


def parse_instance_metrics_for_scene(eval_dir, scene_name):
    per_scene = os.path.join(eval_dir, f"{scene_name}_evaluation_test.txt")
    m = parse_instance_metrics_from_eval(per_scene)
    if m:
        return m

    total_path = os.path.join(eval_dir, "evaluation_total_test.txt")
    if not os.path.exists(total_path):
        return None

    wanted = {
        "Instance Segmentation mPrecision:": "mPrecision",
        "Instance Segmentation mRecall:": "mRecall",
        "Instance Segmentation F1 score:": "F1",
        "Instance Segmentation meanPQ (things):": "meanPQ_things",
        "Instance Segmentation mMUCov:": "mMUCov",
        "Instance Segmentation mMWCov:": "mMWCov",
    }
    out = {}
    in_scene = False
    with open(total_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Scene: "):
                in_scene = (s == f"Scene: {scene_name}")
                continue
            if not in_scene:
                continue
            for prefix, key in wanted.items():
                if s.startswith(prefix):
                    mobj = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", s)
                    if mobj:
                        out[key] = float(mobj.group(1))
            # Stop after we collected all panel metrics for this scene.
            if all(k in out for k in ("F1", "mPrecision", "mRecall", "meanPQ_things")):
                break
    return out if out else None
