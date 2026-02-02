#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FF3D_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
VENV_DIR="${FF3D_DIR}/.venv"
WATERSHED_WORK_DIR="${FF3D_DIR}/work_dirs/cass_watershed_infer"

INPUT_LAS="${1:-${WATERSHED_WORK_DIR}/segmented_trees.las}"
OUTPUT_PLY="${2:-${FF3D_DIR}/data/labeled/plys/test/cass/segmented_trees.ply}"
SCAN_NAME="${3:-segmented_trees}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --quiet laspy

if [[ ! -f "${INPUT_LAS}" ]]; then
  echo "Input LAS not found: ${INPUT_LAS}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PLY}")"

python "${FF3D_DIR}/tools/datasets/las_to_ply.py" \
  "${INPUT_LAS}" \
  "${OUTPUT_PLY}" \
  --semantic 0

TEST_LIST="${FF3D_DIR}/data/splits/original/original_test_list.txt"
grep -qxF "${SCAN_NAME}" "${TEST_LIST}" || echo "${SCAN_NAME}" >> "${TEST_LIST}"

pushd "${FF3D_DIR}" >/dev/null
python tools/datasets/preprocess_dataset.py --test_scan_names_file "${TEST_LIST}"
popd >/dev/null

echo "Done. Added ${SCAN_NAME} and prepared ForAINetV2 test data."
