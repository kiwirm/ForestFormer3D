#!/usr/bin/env python3
"""Assign tree_id to LAS points from polygon GT shapefile."""
import argparse
import os
import sys
from pathlib import Path

import laspy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WATERSHED_DIR = REPO_ROOT / "watershed"
if str(WATERSHED_DIR) not in sys.path:
    sys.path.insert(0, str(WATERSHED_DIR))

from snake_refine import build_snake_surface_from_points, refine_polygons_to_valleys


def _require_geopandas():
    try:
        import geopandas as gpd  # noqa: F401
        return gpd
    except Exception as exc:
        raise RuntimeError(
            "geopandas is required for polygon join. Install with: pip install geopandas"
        ) from exc


def _require_shapely():
    try:
        import shapely  # noqa: F401
        return shapely
    except Exception as exc:
        raise RuntimeError(
            "shapely is required. Install with: pip install shapely"
        ) from exc


def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--las", required=True, help="Input LAS/LAZ")
    parser.add_argument("--gt", required=True, help="GT polygons (shp/geojson)")
    parser.add_argument("--out", required=True, help="Output LAS/LAZ with tree_id")
    parser.add_argument("--tree-id-col", default="tree_id", help="Polygon attribute for tree id")
    parser.add_argument("--chunk", type=int, default=2_000_000, help="Points per chunk")
    parser.add_argument(
        "--method",
        choices=["strtree", "gpd"],
        default="gpd",
        help="Point-in-polygon method (gpd is slower but reliable).",
    )
    parser.add_argument(
        "--target-crs",
        default=None,
        help="CRS to use for polygon reprojection (e.g., EPSG:2193). "
             "If not set, attempt to read CRS from LAS header.",
    )
    parser.add_argument(
        "--snake-refine",
        action="store_true",
        help="Refine polygon edges toward local canopy valleys before point-in-polygon join.",
    )
    parser.add_argument(
        "--snake-out-shp",
        default=None,
        help="Optional output path for the (possibly snake-refined) polygons as a shapefile/GeoPackage.",
    )
    parser.add_argument(
        "--snake-cell-size",
        type=float,
        default=0.2,
        help="Raster cell size (m) for snake valley cost surface.",
    )
    parser.add_argument(
        "--snake-min-height",
        type=float,
        default=0.5,
        help="Minimum normalized height (m) for canopy points in snake surface.",
    )
    parser.add_argument(
        "--snake-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for snake CHM.",
    )
    parser.add_argument(
        "--snake-curvature-stretch",
        type=float,
        default=1.0,
        help="Curvature stretch factor for snake valley surface.",
    )
    parser.add_argument(
        "--snake-iters",
        type=int,
        default=20,
        help="Number of active-contour iterations.",
    )
    parser.add_argument(
        "--snake-search-radius-m",
        type=float,
        default=0.6,
        help="Max movement distance per-vertex normal search (m).",
    )
    parser.add_argument(
        "--snake-samples",
        type=int,
        default=11,
        help="Number of candidate samples per normal search.",
    )
    parser.add_argument(
        "--snake-move-penalty",
        type=float,
        default=0.03,
        help="Quadratic penalty for larger normal offsets.",
    )
    parser.add_argument(
        "--snake-smooth-weight",
        type=float,
        default=0.08,
        help="Penalty encouraging local contour smoothness.",
    )
    parser.add_argument(
        "--snake-relax",
        type=float,
        default=0.7,
        help="Relaxation blend applied each snake iteration.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.las):
        raise FileNotFoundError(args.las)
    if not os.path.exists(args.gt):
        raise FileNotFoundError(args.gt)

    gpd = _require_geopandas()
    shapely = _require_shapely()

    polys = gpd.read_file(args.gt)
    if args.tree_id_col not in polys.columns:
        # Robust fallback for vector files without an explicit tree-id attribute.
        # Prefer common FID/OBJECTID fields; otherwise use 1..N row indices.
        fallback_col = None
        for cand in ("FID", "fid", "OBJECTID", "objectid", "id", "ID"):
            if cand in polys.columns:
                fallback_col = cand
                break
        if fallback_col is not None:
            print(
                f"Warning: '{args.tree_id_col}' not found in polygons; "
                f"using '{fallback_col}' + 1 as tree IDs."
            )
            polys[args.tree_id_col] = polys[fallback_col].astype(np.int64) + 1
        else:
            print(
                f"Warning: '{args.tree_id_col}' not found in polygons and no fallback id field "
                "detected; using row index + 1 as tree IDs."
            )
            polys[args.tree_id_col] = np.arange(1, len(polys) + 1, dtype=np.int64)

    las = laspy.read(args.las)
    target_crs = args.target_crs
    if target_crs is None:
        try:
            crs = las.header.parse_crs()
            if crs is not None:
                target_crs = crs.to_string()
        except Exception:
            target_crs = None

    if target_crs is not None:
        if polys.crs is None:
            polys = polys.set_crs(target_crs)
        elif polys.crs.to_string() != target_crs:
            polys = polys.to_crs(target_crs)
    else:
        # No reprojection possible; warn that CRS mismatch will yield no matches
        if polys.crs is not None:
            print(f"Warning: no target CRS specified; GT CRS is {polys.crs}.")

    if args.snake_refine:
        print("Running snake edge refinement against canopy valley surface...")
        if hasattr(las, "classification"):
            cls = np.asarray(las.classification)
        else:
            cls = np.zeros(len(las.x), dtype=np.uint8)
        surfaces = build_snake_surface_from_points(
            np.asarray(las.x),
            np.asarray(las.y),
            np.asarray(las.z),
            cls,
            cell_size=args.snake_cell_size,
            min_height=args.snake_min_height,
            smooth_sigma=args.snake_smooth_sigma,
            curvature_stretch=args.snake_curvature_stretch,
            curvature_pct_clip=(5.0, 95.0),
        )
        polys, snake_stats = refine_polygons_to_valleys(
            polys,
            surfaces,
            cell_size=args.snake_cell_size,
            n_iters=args.snake_iters,
            search_radius_m=args.snake_search_radius_m,
            n_samples=args.snake_samples,
            move_penalty=args.snake_move_penalty,
            smooth_weight=args.snake_smooth_weight,
            relax=args.snake_relax,
        )
        print(
            "Snake refine done: "
            f"polygons={snake_stats.polygons_refined}/{snake_stats.polygons_seen}, "
            f"rings={snake_stats.rings_refined}, "
            f"mean_shift={snake_stats.mean_vertex_shift_m:.3f}m, "
            f"max_shift={snake_stats.max_vertex_shift_m:.3f}m"
        )
    if args.snake_out_shp:
        out_path = Path(args.snake_out_shp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        polys.to_file(out_path)
        print(f"Wrote polygons to {out_path}")
    n = len(las.x)
    tree_ids = np.zeros(n, dtype=np.uint32)

    # Build structures for spatial queries
    geoms = polys["geometry"].to_list()
    tree_id_vals = polys[args.tree_id_col].to_numpy()
    if args.method == "strtree":
        strtree = shapely.STRtree(geoms)

    # Pre-filter by polygon bbox to reduce work
    minx, miny, maxx, maxy = polys.total_bounds
    bbox_mask = (las.x >= minx) & (las.x <= maxx) & (las.y >= miny) & (las.y <= maxy)
    candidate_idx = np.where(bbox_mask)[0]
    print(f"Candidate points within bbox: {candidate_idx.size} / {n}")

    tqdm = _get_tqdm()
    chunk_iter = range(0, candidate_idx.size, args.chunk)
    if tqdm is not None:
        chunk_iter = tqdm(chunk_iter, desc="Assigning tree_id", unit="chunk")

    for start in chunk_iter:
        end = min(start + args.chunk, candidate_idx.size)
        idx_slice = candidate_idx[start:end]
        xs = las.x[idx_slice]
        ys = las.y[idx_slice]
        if args.method == "strtree":
            pts = shapely.points(xs, ys)
            pairs = strtree.query(pts, predicate="contains")
            if pairs.size == 0:
                print(f"Assigned chunk {start}:{end} (0 matched)")
                continue
            poly_idx = pairs[0]
            pt_idx = pairs[1]
            order = np.argsort(pt_idx)
            pt_idx = pt_idx[order]
            poly_idx = poly_idx[order]
            _, first_pos = np.unique(pt_idx, return_index=True)
            pt_idx = pt_idx[first_pos]
            poly_idx = poly_idx[first_pos]
            tree_ids[idx_slice][pt_idx] = tree_id_vals[poly_idx].astype(np.uint32)
            print(f"Assigned chunk {start}:{end} ({len(pt_idx)} matched)")
        else:
            pts = gpd.GeoDataFrame(
                {"idx": np.arange(0, len(idx_slice), dtype=np.int64)},
                geometry=gpd.points_from_xy(xs, ys),
                crs=polys.crs,
            )
            joined = gpd.sjoin(pts, polys, how="left", predicate="within")
            joined = joined.drop_duplicates(subset="idx")
            matched = joined[args.tree_id_col].to_numpy()
            idxs = joined["idx"].to_numpy()
            valid = ~np.isnan(matched)
            tree_ids[idx_slice[idxs[valid]]] = matched[valid].astype(np.uint32)
            print(f"Assigned chunk {start}:{end} ({valid.sum()} matched)")

    # Write output LAS with tree_id extra dim
    out = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out.header = las.header
    out.x, out.y, out.z = las.x, las.y, las.z
    out.intensity = las.intensity if hasattr(las, "intensity") else None
    out.classification = las.classification if hasattr(las, "classification") else None

    if "tree_id" not in out.point_format.extra_dimension_names:
        out.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.uint32))
    out.tree_id = tree_ids

    out.write(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
