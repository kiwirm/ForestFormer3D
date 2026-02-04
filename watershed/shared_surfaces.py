from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SharedSurfaceBundle:
    chm: np.ndarray
    chm_smooth: np.ndarray
    chm_stretched: np.ndarray
    grad_mag: np.ndarray
    cost_surface: np.ndarray
    canopy_mask: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    nx: int
    ny: int


def _normalize01(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    if not np.any(mask):
        return out

    vals = arr[mask]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        return out
    out[mask] = (arr[mask] - lo) / (hi - lo)
    return out


def minimum_curvature(surface: np.ndarray, cell_size: float) -> np.ndarray:
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


def build_shared_surfaces(
    x_veg: np.ndarray,
    y_veg: np.ndarray,
    hag_veg: np.ndarray,
    *,
    cell_size: float,
    smooth_sigma: float,
    min_height: float,
    curvature_stretch: float,
    curvature_pct_clip: tuple[float, float],
    edge_weight: float = 1.0,
    height_weight: float = 1.0,
) -> SharedSurfaceBundle:
    xmin, ymin = float(x_veg.min()), float(y_veg.min())
    xmax, ymax = float(x_veg.max()), float(y_veg.max())
    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))

    chm = np.full((ny, nx), np.nan, dtype=np.float32)
    ix = np.clip(((x_veg - xmin) / cell_size).astype(np.int32), 0, nx - 1)
    iy = np.clip(((ymax - y_veg) / cell_size).astype(np.int32), 0, ny - 1)

    for i, j, h in zip(iy, ix, hag_veg):
        if np.isnan(chm[i, j]) or h > chm[i, j]:
            chm[i, j] = h

    chm = np.nan_to_num(chm, nan=0.0)
    chm_smooth = gaussian_filter(chm, smooth_sigma)

    min_curv = minimum_curvature(chm_smooth, cell_size)
    clip_low, clip_high = curvature_pct_clip
    low_val = np.percentile(min_curv, clip_low)
    high_val = np.percentile(min_curv, clip_high)
    if high_val <= low_val:
        curv_norm = np.zeros_like(min_curv, dtype=np.float32)
    else:
        curv_clip = np.clip(min_curv, low_val, high_val)
        curv_norm = (curv_clip - low_val) / (high_val - low_val)
    chm_stretched = chm_smooth * (1.0 + curvature_stretch * curv_norm)

    gy, gx = np.gradient(chm_smooth, cell_size, cell_size)
    grad_mag = np.hypot(gx, gy)

    canopy_mask = chm_smooth > min_height
    h_norm = _normalize01(chm_smooth, canopy_mask)
    g_norm = _normalize01(grad_mag, canopy_mask)
    cost_surface = (height_weight * h_norm) - (edge_weight * g_norm)
    if np.any(canopy_mask):
        outside_cost = float(np.max(cost_surface[canopy_mask]) + 1.0)
    else:
        outside_cost = 1.0
    cost_surface = np.where(canopy_mask, cost_surface, outside_cost).astype(np.float32)

    return SharedSurfaceBundle(
        chm=chm,
        chm_smooth=chm_smooth.astype(np.float32),
        chm_stretched=chm_stretched.astype(np.float32),
        grad_mag=grad_mag.astype(np.float32),
        cost_surface=cost_surface,
        canopy_mask=canopy_mask,
        ix=ix,
        iy=iy,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        nx=nx,
        ny=ny,
    )
