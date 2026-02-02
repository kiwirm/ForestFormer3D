import argparse
import laspy
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from scipy.ndimage import gaussian_filter, generic_filter, binary_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.transform import from_origin
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape, mapping, MultiPolygon
import fiona
from tqdm import tqdm

# ---------------------------
# logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Watershed crown segmentation pipeline",
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_LAS = REPO_ROOT / "data/raw/las/cass/cass.segment.crop.las"
DEFAULT_WORK_DIR = REPO_ROOT / "work_dirs/cass_watershed_infer"

parser.add_argument(
    "--input-las",
    default=str(DEFAULT_INPUT_LAS),
    help="Input LAS file path",
)
parser.add_argument(
    "--work-dir",
    default=str(DEFAULT_WORK_DIR),
    help="Output directory for watershed artifacts",
)
parser.add_argument(
    "--crowns-only",
    action="store_true",
    help="Only write crown shapefile (skip CHM, plots, and point assignment)",
)
ARGS = parser.parse_args()
CROWNS_ONLY = ARGS.crowns_only
INPUT_LAS = Path(ARGS.input_las).resolve()
WORK_DIR = Path(ARGS.work_dir).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)


def to_multipolygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    return None


# ---------------------------
# parameters
# ---------------------------
CELL_SIZE = 0.02
MIN_HEIGHT = 0.5
MIN_PEAK_DIST_M = 1.5
SMOOTH_SIGMA = 1.0

CURVATURE_STRETCH = 1.0
CURVATURE_PCT_CLIP = (5.0, 95.0)
CHM_TIF = WORK_DIR / "chm.tif"
CROWN_VECTOR = WORK_DIR / "tree_crowns.shp"
LABEL_SMOOTH_KERNEL_SIZE = 7
GAP_CLOSE_KERNEL_SIZE = 11


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
if not INPUT_LAS.exists():
    raise FileNotFoundError(f"Input LAS not found: {INPUT_LAS}")
logger.info(f"Input LAS: {INPUT_LAS}")
logger.info(f"Output work_dir: {WORK_DIR}")
las = laspy.read(str(INPUT_LAS))
x, y, z = las.x, las.y, las.z
try:
    crs = las.header.parse_crs()
except Exception:
    crs = None

logger.info("Normalizing heights")
ground = las.classification == 2
ground_z = np.interp(z, z[ground], z[ground])  # crude fallback
hag = z - ground_z

veg_mask = hag > MIN_HEIGHT
x, y, hag = x[veg_mask], y[veg_mask], hag[veg_mask]
logger.info(f"{len(hag)} vegetation points above {MIN_HEIGHT} m")

logger.info("Rasterizing CHM")
xmin, ymin = x.min(), y.min()
xmax, ymax = x.max(), y.max()
nx = int(np.ceil((xmax - xmin) / CELL_SIZE))
ny = int(np.ceil((ymax - ymin) / CELL_SIZE))
chm = np.full((ny, nx), np.nan)
ix = np.clip(((x - xmin) / CELL_SIZE).astype(int), 0, nx - 1)
iy = np.clip(((ymax - y) / CELL_SIZE).astype(int), 0, ny - 1)

for i, j, h in tqdm(
    zip(iy, ix, hag),
    total=len(hag),
    desc="Rasterizing CHM",
    unit="pts",
):
    if np.isnan(chm[i, j]) or h > chm[i, j]:
        chm[i, j] = h

chm = np.nan_to_num(chm, nan=0.0)
chm = gaussian_filter(chm, SMOOTH_SIGMA)
logger.info("CHM rasterized and smoothed")

transform = from_origin(xmin, ymax, CELL_SIZE, CELL_SIZE)

if not CROWNS_ONLY:
    if transform is not None:
        logger.info("Writing CHM GeoTIFF")
        chm_tif = chm.astype(np.float32)
        with rasterio.open(
            str(CHM_TIF),
            "w",
            driver="GTiff",
            height=chm_tif.shape[0],
            width=chm_tif.shape[1],
            count=1,
            dtype=chm_tif.dtype,
            crs=crs,
            transform=transform,
            nodata=0.0,
        ) as dst:
            dst.write(chm_tif, 1)
        logger.info(f"CHM GeoTIFF written to {CHM_TIF}")
    else:
        logger.warning("Missing geotransform; skipping CHM GeoTIFF export")

logger.info("Computing minimum curvature image")
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
logger.info("Detecting tree-top markers")
peaks = peak_local_max(
    chm_stretched,
    min_distance=min_peak_dist,
    threshold_abs=MIN_HEIGHT,
)
logger.info(f"{len(peaks)} peaks detected")

markers = np.zeros_like(chm, dtype=np.int32)
for i, (r, c) in enumerate(peaks, start=1):
    markers[r, c] = i

logger.info("Running marker-controlled watershed on stretched CHM")
labels = watershed(-chm_stretched, markers, mask=chm > MIN_HEIGHT)
logger.info(f"{labels.max()} trees segmented")

logger.info("Smoothing label raster with majority filter")


def _mode_filter(values):
    vals = values.astype(np.int32, copy=False)
    return np.bincount(vals).argmax()


labels_smoothed = generic_filter(labels, _mode_filter, size=LABEL_SMOOTH_KERNEL_SIZE)
labels_smoothed = labels_smoothed.astype(labels.dtype, copy=False)
labels_smoothed[chm <= MIN_HEIGHT] = 0
labels = labels_smoothed

logger.info("Closing tiny gaps in label raster")
gap_structure = np.ones((GAP_CLOSE_KERNEL_SIZE, GAP_CLOSE_KERNEL_SIZE), dtype=bool)
canopy_mask = labels > 0
canopy_closed = binary_closing(canopy_mask, structure=gap_structure)
gap_pixels = canopy_closed & ~canopy_mask
if np.any(gap_pixels):
    labels_mode = generic_filter(labels, _mode_filter, size=LABEL_SMOOTH_KERNEL_SIZE)
    labels[gap_pixels] = labels_mode[gap_pixels]
labels[~canopy_closed] = 0

if transform is None:
    logger.warning("Missing geotransform; skipping crown vectorization")
else:
    logger.info("Vectorizing and writing crowns")
    crs_wkt = None
    if crs is not None:
        try:
            crs_wkt = crs.to_wkt()
        except Exception:
            crs_wkt = None
    shapes_iter = rasterio_shapes(
        labels.astype(np.int32),
        mask=labels > 0,
        transform=transform,
    )
    schema = {"geometry": "MultiPolygon", "properties": {"tree_id": "int"}}
    wrote_any = False
    with fiona.open(
        str(CROWN_VECTOR),
        "w",
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=crs_wkt,
    ) as sink:
        for geom, value in tqdm(
            shapes_iter,
            total=int(labels.max()),
            desc="Vectorizing crowns",
            unit="crown",
        ):
            if value == 0:
                continue
            original_geom = shape(geom)
            geom_obj = to_multipolygon(original_geom)
            if geom_obj is None or geom_obj.is_empty:
                continue
            sink.write({
                "geometry": mapping(geom_obj),
                "properties": {"tree_id": int(value)},
            })
            wrote_any = True
    if wrote_any:
        logger.info(f"Crown Shapefile written to {CROWN_VECTOR}")
    else:
        logger.warning("No crown polygons created; skipping Shapefile")

if not CROWNS_ONLY:
    # ---------------------------
    # plots
    # ---------------------------
    logger.info("Plotting CHM, minimum curvature, stretched CHM, and watershed labels")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    im0 = axes[0].imshow(chm, cmap="viridis")
    axes[0].set_title("Canopy Height Model")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    plot_low = np.percentile(min_curv, 2.0)
    plot_high = np.percentile(min_curv, 98.0)
    im1 = axes[1].imshow(min_curv, cmap="coolwarm", vmin=plot_low, vmax=plot_high)
    axes[1].set_title("Minimum Curvature (2-98% clip)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    axes[2].imshow(chm_stretched, cmap="viridis")
    axes[2].scatter(peaks[:, 1], peaks[:, 0], s=15, c="red")
    axes[2].set_title("Stretched CHM + Markers")

    cmap = ListedColormap(plt.cm.tab20.colors)
    axes[3].imshow(labels, cmap=cmap)
    axes[3].set_title("Watershed Tree Crowns")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # assign tree IDs back to points
    # ---------------------------
    logger.info("Assigning tree IDs back to points")
    tree_ids = labels[iy, ix]

    out = laspy.create(
        point_format=las.header.point_format,
        file_version=las.header.version,
    )
    out.header = las.header
    out.x, out.y, out.z = las.x, las.y, las.z
    out.classification = las.classification

    out.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.uint32))
    out.tree_id = np.zeros(len(las), dtype=np.uint32)
    out.tree_id[veg_mask] = tree_ids

    logger.info("Dropping points with tree_id == 0 and ground classification")
    keep = (out.tree_id != 0) & (out.classification != 2)
    out = out[keep]

    out_file = WORK_DIR / "segmented_trees.las"
    out.write(str(out_file))
    logger.info(f"Segmented LAS written to {out_file}")
