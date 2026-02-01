#!/usr/bin/env bash
set -euo pipefail

# End-to-end Cass pipeline: LAS + polygons -> labeled LAS -> tiled PLYs -> derived/bin -> info PKLs

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"

source "$VENV_ACTIVATE"
export PYTHONPATH="$ROOT_DIR"

echo "Cleaning Cass outputs (PLYs, derived arrays, processed bins, infos)..."
rm -rf "$ROOT_DIR/data/labeled/plys/train_val/cass" "$ROOT_DIR/data/labeled/plys/test/cass"
rm -f "$ROOT_DIR/data/derived/instance_data"/cass_*_*.npy
rm -f "$ROOT_DIR/data/derived/infos"/cass_oneformer3d_infos_*.pkl
rm -f "$ROOT_DIR/data/processed/points"/cass_*.bin
rm -f "$ROOT_DIR/data/processed/instance_mask"/cass_*.bin
rm -f "$ROOT_DIR/data/processed/semantic_mask"/cass_*.bin

LAS_INPUT="$ROOT_DIR/data/raw/las/cass/cass.segment.crop.las"
VECTORS_SHAPE="$ROOT_DIR/data/raw/vectors/tree_crowns.shp"
LABELED_LAS="$ROOT_DIR/data/intermediate/cass_labeled.las"

TILE_SIZE=30
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SPLIT_PREFIX=cass
TARGET_CRS="EPSG:2193"
TREE_ID_COL=tree_id

# 1) Assign tree_id from polygons
python tools/datasets/assign_tree_id_from_shp.py \
  --las "$LAS_INPUT" \
  --gt "$VECTORS_SHAPE" \
  --out "$LABELED_LAS" \
  --tree-id-col "$TREE_ID_COL" \
  --target-crs "$TARGET_CRS"

# 2) Tile labeled LAS -> labeled PLYs + split lists
python tools/datasets/las_to_ply_tiles.py \
  "$LABELED_LAS" \
  --tile-size "$TILE_SIZE" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO" \
  --split-prefix "$SPLIT_PREFIX" \
  --splits-dir "$ROOT_DIR/data/splits/cass"

# 3) Build derived arrays / bins
python tools/datasets/preprocess_dataset.py \
  --train_scan_names_file data/splits/cass/cass_train_list.txt \
  --val_scan_names_file data/splits/cass/cass_val_list.txt \
  --test_scan_names_file data/splits/cass/cass_test_list.txt

# 4) Build PKL infos
python tools/prep/build_infos.py cass \
  --extra-tag cass

echo "Done. Cass data is ready for training."
