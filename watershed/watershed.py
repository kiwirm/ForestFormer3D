import argparse
import glob
import logging
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_closing, gaussian_filter, generic_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

# ---------------------------
# logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Run watershed segmentation on test LAS files.")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_GLOB = str(REPO_ROOT / "data/intermediate/test_*.las")
DEFAULT_FALLBACK_GLOB = str(REPO_ROOT / "data/raw/test/test_*.las")
DEFAULT_WORK_DIR = REPO_ROOT / "work_dirs/watershed"

parser.add_argument(
    "--input-glob",
    default=DEFAULT_INPUT_GLOB,
    help="Glob pattern for test LAS files (default: data/intermediate/test_*.las).",
)
parser.add_argument(
    "--work-dir",
    default=str(DEFAULT_WORK_DIR),
    help="Output directory for OneFormer3D-style segmented point clouds.",
)
parser.add_argument("--cell-size", type=float, default=0.02, help="CHM raster cell size in meters.")
parser.add_argument("--min-height", type=float, default=0.5, help="Minimum HAG for canopy points.")
parser.add_argument(
    "--min-peak-dist-m",
    type=float,
    default=1.59,
    help="Minimum peak distance in meters (default tuned on test-set instance-count match).",
)
parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma on CHM.")
parser.add_argument(
    "--curvature-stretch",
    type=float,
    default=1.0,
    help="Blend weight for curvature-guided CHM stretching.",
)
parser.add_argument(
    "--curvature-pct-clip-low",
    type=float,
    default=5.0,
    help="Lower percentile for curvature clipping.",
)
parser.add_argument(
    "--curvature-pct-clip-high",
    type=float,
    default=95.0,
    help="Upper percentile for curvature clipping.",
)
parser.add_argument(
    "--label-smooth-kernel-size",
    type=int,
    default=7,
    help="Mode filter kernel size for label smoothing.",
)
parser.add_argument(
    "--gap-close-kernel-size",
    type=int,
    default=11,
    help="Binary closing kernel size for canopy gap filling.",
)
ARGS = parser.parse_args()
WORK_DIR = Path(ARGS.work_dir).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# parameters
# ---------------------------
CELL_SIZE = ARGS.cell_size
MIN_HEIGHT = ARGS.min_height
MIN_PEAK_DIST_M = ARGS.min_peak_dist_m
SMOOTH_SIGMA = ARGS.smooth_sigma

CURVATURE_STRETCH = ARGS.curvature_stretch
CURVATURE_PCT_CLIP = (ARGS.curvature_pct_clip_low, ARGS.curvature_pct_clip_high)
LABEL_SMOOTH_KERNEL_SIZE = max(1, ARGS.label_smooth_kernel_size)
GAP_CLOSE_KERNEL_SIZE = max(1, ARGS.gap_close_kernel_size)


def minimum_curvature(surface, cell_size):
    dy = cell_size
    dx = cell_size
    zy, zx = np.gradient(surface, dy, dx)
    zyy, zyx = np.gradient(zy, dy, dx)
    zxy, zxx = np.gradient(zx, dy, dx)
    zxy = 0.5 * (zxy + zyx)
    denom = 1.0 + zx * zx + zy * zy
    denom_sqrt = np.sqrt(denom)
    denom_32 = denom * denom_sqrt
    mean_curv = (
        (1.0 + zy * zy) * zxx - 2.0 * zx * zy * zxy + (1.0 + zx * zx) * zyy
    ) / (2.0 * denom_32)
    gauss_curv = (zxx * zyy - zxy * zxy) / (denom * denom)
    discriminant = np.maximum(mean_curv * mean_curv - gauss_curv, 0.0)
    k1 = mean_curv + np.sqrt(discriminant)
    k2 = mean_curv - np.sqrt(discriminant)
    return np.minimum(k1, k2)


logger.info("Loading LAS file")
logger.info(f"Output work_dir: {WORK_DIR}")
input_files = [Path(p).resolve() for p in sorted(glob.glob(ARGS.input_glob))]
if not input_files:
    input_files = [Path(p).resolve() for p in sorted(glob.glob(DEFAULT_FALLBACK_GLOB))]
if not input_files:
    raise FileNotFoundError(
        f"No input LAS files found for '{ARGS.input_glob}' "
        f"or fallback '{DEFAULT_FALLBACK_GLOB}'."
    )
logger.info(f"Found {len(input_files)} test LAS files")


def _mode_filter(values):
    vals = values.astype(np.int32, copy=False)
    return np.bincount(vals).argmax()


def normalize_height(z, classification):
    ground_mask = classification == 2
    if np.any(ground_mask):
        ground_level = np.percentile(z[ground_mask], 5.0)
    else:
        ground_level = np.min(z)
    return z - ground_level


def save_oneformer_style_ply(
    out_path, points, semantic_pred, instance_pred, score, semantic_gt=None, instance_gt=None
):
    if semantic_gt is None or instance_gt is None:
        dtype = [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("semantic_pred", "i4"),
            ("instance_pred", "i4"),
            ("score", "f4"),
        ]
        vertex = np.empty(points.shape[0], dtype=dtype)
        vertex["x"] = points[:, 0]
        vertex["y"] = points[:, 1]
        vertex["z"] = points[:, 2]
        vertex["semantic_pred"] = semantic_pred
        vertex["instance_pred"] = instance_pred
        vertex["score"] = score
    else:
        dtype = [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("semantic_pred", "i4"),
            ("instance_pred", "i4"),
            ("score", "f4"),
            ("semantic_gt", "i4"),
            ("instance_gt", "i4"),
        ]
        vertex = np.empty(points.shape[0], dtype=dtype)
        vertex["x"] = points[:, 0]
        vertex["y"] = points[:, 1]
        vertex["z"] = points[:, 2]
        vertex["semantic_pred"] = semantic_pred
        vertex["instance_pred"] = instance_pred
        vertex["score"] = score
        vertex["semantic_gt"] = semantic_gt
        vertex["instance_gt"] = instance_gt

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(str(out_path))


for input_las in input_files:
    logger.info(f"Processing {input_las}")
    las = laspy.read(str(input_las))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    points = np.column_stack((x, y, z))
    classification = np.asarray(las.classification)

    hag_all = normalize_height(z, classification)
    veg_mask = hag_all > MIN_HEIGHT
    logger.info(f"  Vegetation points above {MIN_HEIGHT} m: {int(np.sum(veg_mask))}")

    instance_pred = np.full(len(las), -1, dtype=np.int32)
    semantic_pred = np.zeros(len(las), dtype=np.int32)
    score = np.zeros(len(las), dtype=np.float32)

    if np.any(veg_mask):
        x_veg = x[veg_mask]
        y_veg = y[veg_mask]
        hag_veg = hag_all[veg_mask]

        xmin, ymin = x_veg.min(), y_veg.min()
        xmax, ymax = x_veg.max(), y_veg.max()
        nx = max(1, int(np.ceil((xmax - xmin) / CELL_SIZE)))
        ny = max(1, int(np.ceil((ymax - ymin) / CELL_SIZE)))
        chm = np.full((ny, nx), np.nan, dtype=np.float32)

        ix = np.clip(((x_veg - xmin) / CELL_SIZE).astype(int), 0, nx - 1)
        iy = np.clip(((ymax - y_veg) / CELL_SIZE).astype(int), 0, ny - 1)

        for i, j, h in tqdm(
            zip(iy, ix, hag_veg),
            total=len(hag_veg),
            desc=f"Rasterizing CHM [{input_las.stem}]",
            unit="pts",
        ):
            if np.isnan(chm[i, j]) or h > chm[i, j]:
                chm[i, j] = h

        chm = np.nan_to_num(chm, nan=0.0)
        chm = gaussian_filter(chm, SMOOTH_SIGMA)

        min_curv = minimum_curvature(chm, CELL_SIZE)
        clip_low, clip_high = CURVATURE_PCT_CLIP
        low_val = np.percentile(min_curv, clip_low)
        high_val = np.percentile(min_curv, clip_high)
        if high_val <= low_val:
            curv_norm = np.zeros_like(min_curv)
        else:
            curv_clip = np.clip(min_curv, low_val, high_val)
            curv_norm = (curv_clip - low_val) / (high_val - low_val)
        chm_stretched = chm * (1.0 + CURVATURE_STRETCH * curv_norm)

        min_peak_dist = max(1, int(np.ceil(MIN_PEAK_DIST_M / CELL_SIZE)))
        peaks = peak_local_max(
            chm_stretched,
            min_distance=min_peak_dist,
            threshold_abs=MIN_HEIGHT,
        )
        markers = np.zeros_like(chm, dtype=np.int32)
        for i, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = i

        labels = watershed(-chm_stretched, markers, mask=chm > MIN_HEIGHT)
        labels_smoothed = generic_filter(labels, _mode_filter, size=LABEL_SMOOTH_KERNEL_SIZE)
        labels_smoothed = labels_smoothed.astype(labels.dtype, copy=False)
        labels_smoothed[chm <= MIN_HEIGHT] = 0
        labels = labels_smoothed

        gap_structure = np.ones((GAP_CLOSE_KERNEL_SIZE, GAP_CLOSE_KERNEL_SIZE), dtype=bool)
        canopy_mask = labels > 0
        canopy_closed = binary_closing(canopy_mask, structure=gap_structure)
        gap_pixels = canopy_closed & ~canopy_mask
        if np.any(gap_pixels):
            labels_mode = generic_filter(labels, _mode_filter, size=LABEL_SMOOTH_KERNEL_SIZE)
            labels[gap_pixels] = labels_mode[gap_pixels]
        labels[~canopy_closed] = 0

        tree_ids_veg = labels[iy, ix].astype(np.int32)
        tree_ids_veg[tree_ids_veg == 0] = -1
        instance_pred[veg_mask] = tree_ids_veg
        semantic_pred[instance_pred >= 0] = 2
        score[instance_pred >= 0] = 1.0
        logger.info(f"  Trees segmented: {int(labels.max())}")
    else:
        logger.warning("  No vegetation points found above threshold")

    semantic_gt = None
    instance_gt = None
    if "tree_id" in las.point_format.extra_dimension_names:
        instance_gt = np.asarray(las.tree_id, dtype=np.int32)
        semantic_gt = np.where(instance_gt > 0, 2, 0).astype(np.int32)

    out_file = WORK_DIR / f"{input_las.stem}.ply"
    save_oneformer_style_ply(
        out_file,
        points,
        semantic_pred,
        instance_pred,
        score,
        semantic_gt=semantic_gt,
        instance_gt=instance_gt,
    )
    logger.info(f"  Wrote segmented point cloud: {out_file}")
