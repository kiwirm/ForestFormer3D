#!/usr/bin/env python3
"""Active-contour-style polygon edge refinement against canopy valley cost surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

try:
    from shared_surfaces import SharedSurfaceBundle, build_shared_surfaces
except ModuleNotFoundError:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from shared_surfaces import SharedSurfaceBundle, build_shared_surfaces


@dataclass
class SnakeRefineStats:
    polygons_seen: int = 0
    polygons_refined: int = 0
    rings_refined: int = 0
    mean_vertex_shift_m: float = 0.0
    max_vertex_shift_m: float = 0.0


def normalize_height(z: np.ndarray, classification: np.ndarray) -> np.ndarray:
    ground_mask = classification == 2
    if np.any(ground_mask):
        ground_level = np.percentile(z[ground_mask], 5.0)
    else:
        ground_level = np.min(z)
    return z - ground_level


def build_snake_surface_from_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    *,
    cell_size: float,
    min_height: float,
    smooth_sigma: float,
    curvature_stretch: float,
    curvature_pct_clip: tuple[float, float],
) -> SharedSurfaceBundle:
    hag_all = normalize_height(z, classification)
    veg_mask = hag_all > min_height
    if not np.any(veg_mask):
        raise ValueError(f"No vegetation points above min_height={min_height} m.")
    return build_shared_surfaces(
        x[veg_mask],
        y[veg_mask],
        hag_all[veg_mask],
        cell_size=cell_size,
        smooth_sigma=smooth_sigma,
        min_height=min_height,
        curvature_stretch=curvature_stretch,
        curvature_pct_clip=curvature_pct_clip,
    )


def _bilinear_cost(cost: np.ndarray, rf: np.ndarray, cf: np.ndarray) -> np.ndarray:
    ny, nx = cost.shape
    rf = np.clip(rf, 0.0, ny - 1.0)
    cf = np.clip(cf, 0.0, nx - 1.0)

    r0 = np.floor(rf).astype(np.int32)
    c0 = np.floor(cf).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, ny - 1)
    c1 = np.clip(c0 + 1, 0, nx - 1)

    fr = rf - r0
    fc = cf - c0

    v00 = cost[r0, c0]
    v01 = cost[r0, c1]
    v10 = cost[r1, c0]
    v11 = cost[r1, c1]

    return (
        (1.0 - fr) * (1.0 - fc) * v00
        + (1.0 - fr) * fc * v01
        + fr * (1.0 - fc) * v10
        + fr * fc * v11
    )


def _world_to_raster(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    xmin: float,
    ymax: float,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    cf = (xs - xmin) / cell_size
    rf = (ymax - ys) / cell_size
    return rf, cf


def _ring_to_xy(ring) -> np.ndarray:
    coords = np.asarray(ring.coords, dtype=np.float64)
    if len(coords) < 4:
        return np.empty((0, 2), dtype=np.float64)
    xy = coords[:, :2]
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    return xy


def _refine_ring_xy(
    ring_xy: np.ndarray,
    *,
    cost_surface: np.ndarray,
    xmin: float,
    ymax: float,
    cell_size: float,
    n_iters: int,
    search_radius_m: float,
    n_samples: int,
    move_penalty: float,
    smooth_weight: float,
    relax: float,
) -> tuple[np.ndarray, np.ndarray]:
    verts = ring_xy.copy()
    start = verts.copy()
    n = len(verts)
    if n < 3:
        return verts, np.zeros((n,), dtype=np.float64)

    offsets = np.linspace(-search_radius_m, search_radius_m, n_samples, dtype=np.float64)
    offset_norm = offsets / max(search_radius_m, 1e-6)
    move_reg = move_penalty * (offset_norm * offset_norm)

    for _ in range(n_iters):
        proposed = verts.copy()
        for i in range(n):
            prev_v = verts[(i - 1) % n]
            curr_v = verts[i]
            next_v = verts[(i + 1) % n]

            tangent = next_v - prev_v
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-9:
                continue

            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64) / tnorm
            candidates = curr_v[None, :] + offsets[:, None] * normal[None, :]

            rf, cf = _world_to_raster(
                candidates[:, 0],
                candidates[:, 1],
                xmin=xmin,
                ymax=ymax,
                cell_size=cell_size,
            )
            data_term = _bilinear_cost(cost_surface, rf, cf)

            smooth_target = 0.5 * (prev_v + next_v)
            smooth_term = smooth_weight * np.sum((candidates - smooth_target[None, :]) ** 2, axis=1)

            total = data_term + move_reg + smooth_term
            proposed[i] = candidates[int(np.argmin(total))]

        verts = (1.0 - relax) * verts + relax * proposed

    shifts = np.linalg.norm(verts - start, axis=1)
    return verts, shifts


def refine_polygons_to_valleys(
    polygons_gdf,
    surfaces: SharedSurfaceBundle,
    *,
    cell_size: float,
    n_iters: int = 20,
    search_radius_m: float = 0.5,
    n_samples: int = 11,
    move_penalty: float = 0.03,
    smooth_weight: float = 0.08,
    relax: float = 0.7,
):
    import shapely.geometry as sgeom

    if n_samples < 3:
        raise ValueError("n_samples must be >= 3")
    if search_radius_m <= 0:
        raise ValueError("search_radius_m must be > 0")

    refined = polygons_gdf.copy()
    stats = SnakeRefineStats()
    all_shifts = []

    new_geoms = []
    for geom in refined.geometry:
        stats.polygons_seen += 1
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            new_geoms.append(geom)
            continue
        rebuilt_parts = []
        changed_any = False

        for poly in polys:
            shell_xy = _ring_to_xy(poly.exterior)
            if shell_xy.size == 0:
                rebuilt_parts.append(poly)
                continue
            shell_new, shell_shift = _refine_ring_xy(
                shell_xy,
                cost_surface=surfaces.cost_surface,
                xmin=surfaces.xmin,
                ymax=surfaces.ymax,
                cell_size=cell_size,
                n_iters=n_iters,
                search_radius_m=search_radius_m,
                n_samples=n_samples,
                move_penalty=move_penalty,
                smooth_weight=smooth_weight,
                relax=relax,
            )
            rings = [np.vstack([shell_new, shell_new[0]])]
            ring_shifts = [shell_shift]

            for interior in poly.interiors:
                ring_xy = _ring_to_xy(interior)
                if ring_xy.size == 0:
                    continue
                ring_new, ring_shift = _refine_ring_xy(
                    ring_xy,
                    cost_surface=surfaces.cost_surface,
                    xmin=surfaces.xmin,
                    ymax=surfaces.ymax,
                    cell_size=cell_size,
                    n_iters=n_iters,
                    search_radius_m=search_radius_m,
                    n_samples=n_samples,
                    move_penalty=move_penalty,
                    smooth_weight=smooth_weight,
                    relax=relax,
                )
                rings.append(np.vstack([ring_new, ring_new[0]]))
                ring_shifts.append(ring_shift)

            new_poly = sgeom.Polygon(rings[0], [r for r in rings[1:]])
            if (not new_poly.is_valid) or new_poly.is_empty:
                fixed = new_poly.buffer(0)
                if fixed.is_empty:
                    rebuilt_parts.append(poly)
                    continue
                if fixed.geom_type == "Polygon":
                    new_poly = fixed
                elif fixed.geom_type == "MultiPolygon":
                    new_poly = max(fixed.geoms, key=lambda g: g.area)
                else:
                    rebuilt_parts.append(poly)
                    continue

            changed_any = True
            stats.rings_refined += len(rings)
            for rs in ring_shifts:
                all_shifts.append(rs)
            rebuilt_parts.append(new_poly)

        if len(rebuilt_parts) == 1:
            rebuilt = rebuilt_parts[0]
        else:
            rebuilt = sgeom.MultiPolygon(rebuilt_parts)

        if changed_any:
            stats.polygons_refined += 1
        new_geoms.append(rebuilt)

    if all_shifts:
        shift_cat = np.concatenate(all_shifts)
        stats.mean_vertex_shift_m = float(np.mean(shift_cat))
        stats.max_vertex_shift_m = float(np.max(shift_cat))

    refined = refined.set_geometry(new_geoms)
    return refined, stats
