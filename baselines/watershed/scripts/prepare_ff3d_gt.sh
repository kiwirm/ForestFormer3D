#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FF3D_DIR="${ROOT_DIR}/../forestformer3d"
VENV_DIR="${ROOT_DIR}/.venv"

INPUT_LAS="${1:-${ROOT_DIR}/segmented_trees.las}"
OUTPUT_PLY="${2:-${FF3D_DIR}/data/raw/plys/test/segmented_trees.ply}"
SCAN_NAME="${3:-segmented_trees}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --quiet laspy

python "${FF3D_DIR}/data/ForAINetV2/las_to_ply.py" \
  "${INPUT_LAS}" \
  "${OUTPUT_PLY}" \
  --semantic 0

TEST_LIST="${FF3D_DIR}/data/ForAINetV2/meta_data/test_list.txt"
grep -qxF "${SCAN_NAME}" "${TEST_LIST}" || echo "${SCAN_NAME}" >> "${TEST_LIST}"

pushd "${FF3D_DIR}/data/ForAINetV2" >/dev/null
python batch_load_ForAINetV2_data.py --test_scan_names_file meta_data/test_list.txt
popd >/dev/null

echo "Done. Added ${SCAN_NAME} and prepared ForAINetV2 test data."
