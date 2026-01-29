#!/usr/bin/env python3
"""Unified dataset prep for XYZ or XYZ+RGB+I."""
import argparse
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(DATA, "splits")


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["xyz", "rgb"], required=True)
    parser.add_argument("--las", required=True, help="Input LAS/LAZ")
    parser.add_argument("--out-ply", required=True, help="Output PLY path")
    parser.add_argument("--semantic", type=int, default=0)
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--include-intensity", action="store_true")
    parser.add_argument("--normalize-intensity", action="store_true")
    parser.add_argument("--scan-name", required=True, help="Base name in splits/test_list.txt")
    args = parser.parse_args()

    las_to_ply = os.path.join(ROOT, "tools", "datasets", "las_to_ply.py")
    if args.mode == "rgb":
        args.include_rgb = True
        args.include_intensity = True

    cmd = ["python", las_to_ply, args.las, args.out_ply, "--semantic", str(args.semantic)]
    if args.include_rgb:
        cmd.append("--include-rgb")
    if args.include_intensity:
        cmd.append("--include-intensity")
    if args.normalize_intensity:
        cmd.append("--normalize-intensity")
    run(cmd)

    # ensure scan listed in test split
    test_list = os.path.join(SPLITS, "test_list.txt")
    with open(test_list, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if args.scan_name not in lines:
        with open(test_list, "a", encoding="utf-8") as f:
            f.write(args.scan_name + "\n")

    # batch load
    batch_script = os.path.join(ROOT, "tools", "datasets", f"batch_load_ForAINetV2_data{'' if args.mode=='xyz' else '_rgb'}.py")
    run(["python", batch_script, "--test_scan_names_file", test_list])

    # create info PKLs
    create_script = os.path.join(ROOT, "tools", f"create_data_forainetv2{'' if args.mode=='xyz' else '_rgb'}.py")
    out_dir = os.path.join(DATA, "derived", "infos")
    run(["python", create_script, f"forainetv2{'' if args.mode=='xyz' else '_rgb'}", "--root-path", DATA, "--out-dir", out_dir])


if __name__ == "__main__":
    main()
