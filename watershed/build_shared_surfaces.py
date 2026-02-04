import argparse
import glob
from pathlib import Path

import laspy
import numpy as np

from shared_surfaces import build_shared_surfaces


def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def normalize_height(z: np.ndarray, classification: np.ndarray) -> np.ndarray:
    ground_mask = classification == 2
    if np.any(ground_mask):
        ground_level = np.percentile(z[ground_mask], 5.0)
    else:
        ground_level = np.min(z)
    return z - ground_level


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build shared canopy rasters used by watershed and XY-conflation "
            "(CHM, gradient, and boundary-aware cost surface)."
        )
    )
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser.add_argument(
        "--input-glob",
        default=str(repo_root / "data/intermediate/*.las"),
        help="Input LAS/LAZ glob.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "data/derived/crown_surfaces"),
        help="Directory where per-scene .npz surface bundles are written.",
    )
    parser.add_argument("--cell-size", type=float, default=0.02, help="Raster cell size (m).")
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.5,
        help="Minimum normalized height considered canopy (m).",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for CHM.",
    )
    parser.add_argument(
        "--curvature-stretch",
        type=float,
        default=1.0,
        help="Curvature-guided CHM stretching factor (shared with watershed).",
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
        "--height-weight",
        type=float,
        default=1.0,
        help="Weight for normalized CHM term in the cost surface.",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=1.0,
        help="Weight for normalized gradient term in the cost surface.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_files = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not input_files:
        raise FileNotFoundError(f"No LAS files matched: {args.input_glob}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tqdm = _get_tqdm()
    file_iter = input_files
    if tqdm is not None:
        file_iter = tqdm(input_files, desc="Building shared surfaces", unit="file")

    for input_las in file_iter:
        las = laspy.read(str(input_las))
        x = np.asarray(las.x)
        y = np.asarray(las.y)
        z = np.asarray(las.z)
        classification = np.asarray(las.classification)
        hag_all = normalize_height(z, classification)
        veg_mask = hag_all > args.min_height
        if not np.any(veg_mask):
            print(f"{input_las.name}: skipped (no canopy points above {args.min_height} m)")
            continue

        surfaces = build_shared_surfaces(
            x[veg_mask],
            y[veg_mask],
            hag_all[veg_mask],
            cell_size=args.cell_size,
            smooth_sigma=args.smooth_sigma,
            min_height=args.min_height,
            curvature_stretch=args.curvature_stretch,
            curvature_pct_clip=(args.curvature_pct_clip_low, args.curvature_pct_clip_high),
            edge_weight=args.edge_weight,
            height_weight=args.height_weight,
        )
        out_path = out_dir / f"{input_las.stem}_surfaces.npz"
        np.savez_compressed(
            out_path,
            chm=surfaces.chm,
            chm_smooth=surfaces.chm_smooth,
            chm_stretched=surfaces.chm_stretched,
            grad_mag=surfaces.grad_mag,
            cost_surface=surfaces.cost_surface,
            canopy_mask=surfaces.canopy_mask.astype(np.uint8),
            xmin=surfaces.xmin,
            ymin=surfaces.ymin,
            xmax=surfaces.xmax,
            ymax=surfaces.ymax,
            nx=surfaces.nx,
            ny=surfaces.ny,
            cell_size=float(args.cell_size),
            min_height=float(args.min_height),
            edge_weight=float(args.edge_weight),
            height_weight=float(args.height_weight),
        )
        print(f"{input_las.name}: wrote {out_path}")


if __name__ == "__main__":
    main()
