import argparse
import glob
from pathlib import Path

import laspy
import numpy as np


def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Keep only locally high points in each XY bin of LAS files. "
            "Useful for crown-focused training by suppressing lower-canopy/trunk/ground clutter."
        )
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for input LAS/LAZ files, e.g. 'data/intermediate/*.las'",
    )
    parser.add_argument(
        "--output-dir",
        default="data/filtered",
        help="Output directory for filtered LAS/LAZ files.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.5,
        help="XY bin size in the same units as LAS coordinates (default: 0.5).",
    )
    parser.add_argument(
        "--keep-top-percent",
        type=float,
        default=15.0,
        help="Percent of highest-Z points to keep per bin (default: 15).",
    )
    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=20,
        help=(
            "Bins with fewer points than this are handled by --small-bin-policy "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--small-bin-policy",
        choices=["keep_all", "drop_all"],
        default="keep_all",
        help="How to handle sparse bins (default: keep_all).",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=None,
        help="If set, drop points with z < min-z before local percentile filtering.",
    )
    return parser.parse_args()


def local_top_percentile_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bin_size: float,
    keep_top_percent: float,
    min_points_per_bin: int,
    small_bin_policy: str,
) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    x0 = np.min(x)
    y0 = np.min(y)
    ix = np.floor((x - x0) / bin_size).astype(np.int64)
    iy = np.floor((y - y0) / bin_size).astype(np.int64)
    bins = np.column_stack([ix, iy])

    _, inv, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)

    order = np.argsort(inv)
    inv_sorted = inv[order]
    starts = np.r_[0, np.flatnonzero(np.diff(inv_sorted)) + 1]
    ends = np.r_[starts[1:], inv_sorted.size]

    keep = np.zeros(x.size, dtype=bool)
    q = max(0.0, min(100.0, 100.0 - keep_top_percent))

    tqdm = _get_tqdm()
    bin_iter = zip(starts, ends)
    if tqdm is not None:
        bin_iter = tqdm(
            bin_iter,
            total=len(starts),
            desc="Filtering bins",
            unit="bin",
            leave=False,
        )

    for start, end in bin_iter:
        idx = order[start:end]
        bin_id = inv_sorted[start]
        c = counts[bin_id]
        if c < min_points_per_bin:
            if small_bin_policy == "keep_all":
                keep[idx] = True
            continue

        zz = z[idx]
        thr = np.percentile(zz, q)
        keep[idx] = zz >= thr

    return keep


def filter_file(
    in_path: Path,
    out_path: Path,
    bin_size: float,
    keep_top_percent: float,
    min_points_per_bin: int,
    small_bin_policy: str,
    min_z: float | None,
):
    las = laspy.read(str(in_path))
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    pre_keep = np.ones(x.size, dtype=bool)
    if min_z is not None:
        pre_keep = z >= min_z
        x = x[pre_keep]
        y = y[pre_keep]
        z = z[pre_keep]

    keep = local_top_percentile_mask(
        x=x,
        y=y,
        z=z,
        bin_size=bin_size,
        keep_top_percent=keep_top_percent,
        min_points_per_bin=min_points_per_bin,
        small_bin_policy=small_bin_policy,
    )

    out_las = laspy.LasData(las.header)
    if min_z is not None:
        full_keep = np.zeros(len(las.points), dtype=bool)
        kept_indices = np.flatnonzero(pre_keep)
        full_keep[kept_indices] = keep
    else:
        full_keep = keep
    out_las.points = las.points[full_keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_las.write(str(out_path))

    n_in = int(x.size)
    n_total_in = len(las.points)
    n_post_minz = int(np.count_nonzero(pre_keep)) if min_z is not None else n_total_in
    n_out = int(np.count_nonzero(full_keep))
    ratio = (n_out / n_total_in) if n_total_in > 0 else 0.0
    if min_z is not None:
        print(
            f"{in_path.name}: min-z kept {n_post_minz}/{n_total_in} points, "
            f"local-percentile kept {n_out}/{n_total_in} total ({ratio:.2%}), wrote {out_path}"
        )
        return
    print(
        f"{in_path.name}: kept {n_out}/{n_total_in} points "
        f"({ratio:.2%}), wrote {out_path}"
    )


def main():
    args = parse_args()
    in_files = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not in_files:
        raise FileNotFoundError(f"No files matched --input-glob: {args.input_glob}")

    out_dir = Path(args.output_dir)
    tqdm = _get_tqdm()
    file_iter = in_files
    if tqdm is not None:
        file_iter = tqdm(in_files, desc="Processing files", unit="file")

    for in_path in file_iter:
        out_path = out_dir / in_path.name
        filter_file(
            in_path=in_path,
            out_path=out_path,
            bin_size=args.bin_size,
            keep_top_percent=args.keep_top_percent,
            min_points_per_bin=args.min_points_per_bin,
            small_bin_policy=args.small_bin_policy,
            min_z=args.min_z,
        )


if __name__ == "__main__":
    main()
