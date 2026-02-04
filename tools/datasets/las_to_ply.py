import argparse
import os

import laspy
import numpy as np

from tools.support.plyutils import write_ply


def _get_rgb(las, n_points):
    if not hasattr(las, "red") or not hasattr(las, "green") or not hasattr(las, "blue"):
        return np.zeros((n_points, 3), dtype=np.float32)
    rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
    # Convert 16-bit LAS colors to 0-255 range if needed
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAS/LAZ with tree_id to labeled PLY."
    )
    parser.add_argument("input_las", help="Path to input LAS/LAZ file")
    parser.add_argument("output_ply", help="Path to output PLY file")
    parser.add_argument(
        "--semantic",
        type=int,
        default=None,
        help="Override semantic_seg with a constant. If omitted, derive from tree_id.",
    )
    parser.add_argument(
        "--tree-id-dim",
        default="tree_id",
        help="Extra bytes dimension name for instance ids (default: tree_id)",
    )
    parser.add_argument(
        "--normalize-intensity",
        action="store_true",
        help="Normalize intensity to [0,1]",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_las):
        raise FileNotFoundError(f"Input LAS not found: {args.input_las}")

    las = laspy.read(args.input_las)

    if args.tree_id_dim not in las.point_format.extra_dimension_names:
        raise ValueError(
            f"'{args.tree_id_dim}' not found in LAS extra dimensions. "
            f"Available: {las.point_format.extra_dimension_names}"
        )

    # Keep coordinates in float64 to preserve centimeter-level precision
    # at large UTM magnitudes (float32 can quantize to ~0.5m here).
    points = np.vstack((las.x, las.y, las.z)).T
    n_points = points.shape[0]
    tree_ids = np.asarray(getattr(las, args.tree_id_dim), dtype=np.int32)
    if args.semantic is None:
        # Derive semantics: ground=0, trees=3 (leaf after -1 mapping)
        semantic_seg = np.zeros_like(tree_ids, dtype=np.int32)
        semantic_seg[tree_ids > 0] = 3
    else:
        semantic_seg = np.full(tree_ids.shape, args.semantic, dtype=np.int32)

    fields = [points]
    field_names = ["x", "y", "z"]

    rgb = _get_rgb(las, n_points)
    fields.append(rgb)
    field_names += ["red", "green", "blue"]

    intensity = _get_intensity(las, n_points, args.normalize_intensity)
    fields.append(intensity)
    field_names += ["intensity"]

    fields += [semantic_seg, tree_ids]
    field_names += ["semantic_seg", "treeID"]

    ok = write_ply(
        args.output_ply,
        fields,
        field_names,
    )
    if not ok:
        raise RuntimeError("Failed to write PLY.")

    print(f"Wrote {args.output_ply} with {points.shape[0]} points.")


if __name__ == "__main__":
    main()
