# ForestFormer3D Single-Scene Runbook (Instance-Only)

This runbook documents the current single-scene workflow with RGB+I data prep and instance-only training.
Assumes a CUDA Linux GPU machine and `.venv` is set up.

## Current naming convention

Configs and work dirs match by name:
- `configs/pretrained.py` -> `work_dirs/pretrained`
- `configs/xyz.py` -> `work_dirs/xyz`
- `configs/xyzrgb.py` -> `work_dirs/xyzrgb`

All three configs currently use:
- `max_epochs=3000`
- `val_interval=100`

## Single-scene data pipeline

Run the pipeline:

```bash
source .venv/bin/activate
export PYTHONPATH=.
./tools/run_prep
```

What it does:
- Uses `data/raw/las/cass/cass.segment.crop.train.las` as train scene input.
- Uses `data/raw/las/cass/cass.segment.crop.test.las` as test scene input only if the file exists.
- Builds one PLY scene for train and one for test (no tiles):
  - `data/labeled/plys/train_val/train_scene.ply`
  - `data/labeled/plys/test/test_scene.ply` (optional)
- Writes split files:
  - `data/splits/scene/scene_train_list.txt`
  - `data/splits/scene/scene_val_list.txt`
  - `data/splits/scene/scene_test_list.txt`
- Rebuilds derived caches and info files:
  - `data/derived/infos/scene_oneformer3d_infos_train.pkl`
  - `data/derived/infos/scene_oneformer3d_infos_val.pkl`
  - `data/derived/infos/scene_oneformer3d_infos_test.pkl`

The pipeline also auto-creates required folders if missing (`data/derived`, `data/processed`, `data/splits/scene`, `data/labeled/plys/...`, `data/intermediate`).

## Workflows

### 1) Pretrained inference -> eval

```bash
source .venv/bin/activate
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python tools/training/test.py \
  configs/pretrained.py \
  data/models/epoch_3000_fix_spconv.pth \
  --cfg-options test_cfg.output_dir=work_dirs/pretrained/eval

python tools/evaluation/final_eval.py work_dirs/pretrained/eval
```

### 2) Train XYZ -> inference -> eval

```bash
source .venv/bin/activate
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python tools/training/train.py \
  configs/xyz.py \
  --work-dir work_dirs/xyz

CUDA_VISIBLE_DEVICES=0 python tools/training/test.py \
  configs/xyz.py \
  work_dirs/xyz/latest.pth \
  --cfg-options test_cfg.output_dir=work_dirs/xyz/eval

python tools/evaluation/final_eval.py work_dirs/xyz/eval
```

Resume training in the same dir:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/training/train.py \
  configs/xyz.py \
  --work-dir work_dirs/xyz \
  --resume
```

### 3) Train XYZRGB -> inference -> eval

```bash
source .venv/bin/activate
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python tools/training/train.py \
  configs/xyzrgb.py \
  --work-dir work_dirs/xyzrgb

CUDA_VISIBLE_DEVICES=0 python tools/training/test.py \
  configs/xyzrgb.py \
  work_dirs/xyzrgb/latest.pth \
  --cfg-options test_cfg.output_dir=work_dirs/xyzrgb/eval

python tools/evaluation/final_eval.py work_dirs/xyzrgb/eval
```

## Notes

- `configs/xyz.py` uses `use_dim=[0,1,2]` (XYZ only).
- `configs/xyzrgb.py` uses `use_dim=[0,1,2,3,4,5,6]` (XYZ+RGB+I).
- Both configs use anisotropic voxelization for crown-focused training: `voxel_size=[0.16, 0.16, 0.28]` with `GridSample(grid_size=0.16)`.
- Both configs use sliding-cylinder merge for inference (`sliding_inference=True`, `radius=16`, `stride=4`, edge buffer `0.5`, overlap merge).
- The pretrained checkpoint is semantic+instance and is not directly comparable to instance-only runs.
