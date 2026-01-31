#!/usr/bin/env bash
set -euo pipefail

# Package minimal artifacts needed to train on a prepared dataset.
# Usage: ./tools/package_training_data.sh [cass|original|combined] [output_tar]

DATASET=${1:-cass}
OUT_TAR=${2:-training_package_${DATASET}.tar.gz}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

INFO_DIR="data/derived/infos"
SPLIT_DIR="data/splits/${DATASET}"
INSTANCE_DIR="data/derived/instance_data"
PROCESSED_DIR="data/processed"

if [[ ! -d "$SPLIT_DIR" ]]; then
  echo "Missing split dir: $SPLIT_DIR" >&2
  exit 1
fi
if [[ ! -d "$INSTANCE_DIR" ]]; then
  echo "Missing instance cache: $INSTANCE_DIR" >&2
  exit 1
fi
if [[ ! -d "$PROCESSED_DIR" ]]; then
  echo "Missing processed bins: $PROCESSED_DIR" >&2
  exit 1
fi

INFO_FILES=(
  "$INFO_DIR/${DATASET}_oneformer3d_infos_train.pkl"
  "$INFO_DIR/${DATASET}_oneformer3d_infos_val.pkl"
  "$INFO_DIR/${DATASET}_oneformer3d_infos_test.pkl"
)
for f in "${INFO_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing info PKL: $f" >&2
    exit 1
  fi
 done

# Data needed for training
DATA_PATHS=(
  "$SPLIT_DIR"
  "$INSTANCE_DIR"
  "$PROCESSED_DIR"
  "data/models"
  "${INFO_FILES[@]}"
)

# Create archive
printf "Packaging to %s...\n" "$OUT_TAR"

tar -czf "$OUT_TAR" \
  "${DATA_PATHS[@]}"

printf "Done: %s\n" "$OUT_TAR"
