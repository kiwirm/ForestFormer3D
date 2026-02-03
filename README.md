# ForestFormer3D Cass Runbook (Instance-Only)

This runbook documents the current Cass workflow with RGB+I data prep and instance-only training.
Assumes a CUDA Linux GPU machine and `.venv` is set up.

## Current naming convention

Configs and work dirs now match by name:
- `configs/pretrained.py` -> `work_dirs/pretrained`
- `configs/xyz.py` -> `work_dirs/xyz`
- `configs/xyzrgb.py` -> `work_dirs/xyzrgb`

## Epoch defaults (important)

- `configs/xyz.py` trains for `max_epochs=100` by default.
- `configs/xyzrgb.py` trains for `max_epochs=100` by default.
- `configs/pretrained.py` uses `max_epochs=3000` (pretrained config behavior).

To run XYZ for 3000 epochs without editing the config:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/training/train.py \
  configs/xyz.py \
  --work-dir work_dirs/xyz \
  --cfg-options train_cfg.max_epochs=3000 train_cfg.val_interval=100
```

## Current split setup (important)

The Cass split files are intentionally set so test data is also used for training:
- `data/splits/cass/cass_train_list.txt` (65 tiles)
- `data/splits/cass/cass_val_list.txt` (65 tiles)
- `data/splits/cass/cass_test_list.txt` (65 tiles)

All three info files were rebuilt from these lists:
- `data/derived/infos/cass_oneformer3d_infos_train.pkl`
- `data/derived/infos/cass_oneformer3d_infos_val.pkl`
- `data/derived/infos/cass_oneformer3d_infos_test.pkl`

## Prepare / rebuild data caches and infos

```bash
source .venv/bin/activate
export PYTHONPATH=.

python tools/datasets/preprocess_dataset.py \
  --train_scan_names_file data/splits/cass/cass_train_list.txt \
  --val_scan_names_file data/splits/cass/cass_val_list.txt \
  --test_scan_names_file data/splits/cass/cass_test_list.txt

python tools/prep/build_infos.py cass --extra-tag cass
```

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

- XYZ config uses `use_dim=[0,1,2]` and ignores RGB+I features at train/test time.
- For crown-focused training, both `configs/xyz.py` and `configs/xyzrgb.py` use anisotropic sampling/voxelization (XY finer than Z): `voxel_size=[0.16, 0.16, 0.28]` with `GridSample(grid_size=0.16)`.
- For full-paper inference behavior, both configs now use sliding-cylinder merge in `model.test_cfg` (`sliding_inference=True`, `radius=16`, `stride=4`, edge buffer `0.5`, score-based overlap merge).
- The pretrained checkpoint is semantic+instance and is not directly comparable to instance-only runs.
- If you want different output locations, override `test_cfg.output_dir` in the test command.
