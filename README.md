# ForestFormer3D Runbook (Current Repo State)

This README reflects the current scripts/configs in this repository.

## 1) Environment (recommended for every run)

```bash
source .venv/bin/activate

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

# Prevent CPU thread oversubscription in dataloader workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## 2) Data preparation (single-scene pipeline)

Run:

```bash
./tools/run_prep
```

`tools/run_prep` currently:
- reads train LAS from `data/raw/train/train.las`
- optionally reads test LAS from `data/raw/test/test.las`
- reads polygons from `data/raw/train/train.shp`
- writes labeled LAS into `data/intermediate/`
- writes PLYs into `data/labeled/train/` and `data/labeled/test/`
- regenerates processed bins under `data/processed/`
- regenerates infos under `data/derived/infos/train.pkl` and `data/derived/infos/test.pkl`

Optional overrides (if needed):
- `TRAIN_LAS_INPUT`
- `TEST_LAS_INPUT`
- `VECTORS_SHAPE`
- `TRAIN_LABELED_LAS`
- `TEST_LABELED_LAS`
- `TARGET_CRS`
- `TREE_ID_COL`
- `SNAKE_REFINE=1` (optional: refine crown polygon edges into local canopy valleys before labeling)
- `SNAKE_CELL_SIZE` (default `0.2`)
- `SNAKE_SEARCH_RADIUS_M` (default `0.6`)
- `SNAKE_ITERS` (default `20`)
- `SNAKE_OUT_SHP` (default `data/raw/train_snake.shp`; written from train pass when snake refine is enabled)

## 3) Training

### Train XYZ

```bash
python tools/training/train.py configs/xyz.py --work-dir work_dirs/xyz
```

Resume:

```bash
python tools/training/train.py configs/xyz.py --work-dir work_dirs/xyz --resume
```

### Run XYZ training in background

```bash
mkdir -p work_dirs/xyz

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nohup python tools/training/train.py \
  configs/xyz.py \
  --work-dir work_dirs/xyz \
  --resume \
  > work_dirs/xyz/train.log 2>&1 &
echo $! > work_dirs/xyz/train.pid
```

Check status/logs:

```bash
ps -p "$(cat work_dirs/xyz/train.pid)" -o pid=,cmd=
tail -f work_dirs/xyz/train.log
```

Stop:

```bash
kill "$(cat work_dirs/xyz/train.pid)"
```

### Train XYZRGB

```bash
python tools/training/train.py configs/xyzrgb.py --work-dir work_dirs/xyzrgb
```

Resume:

```bash
python tools/training/train.py configs/xyzrgb.py --work-dir work_dirs/xyzrgb --resume
```

## 4) Inference + evaluation

### Shared crown preprocessing (watershed + XY-conflation)

Build a shared raster bundle (`CHM`, `CHM gradient`, boundary-aware `cost_surface`) that
both watershed and active-contour style XY line sliding can use:

```bash
python watershed/build_shared_surfaces.py \
  --input-glob 'data/intermediate/test_*.las' \
  --output-dir data/derived/crown_surfaces
```

Each output `.npz` includes:
- `chm`, `chm_smooth`, `chm_stretched`
- `grad_mag`
- `cost_surface` (low CHM + high gradient => lower cost near crown boundaries)
- raster georeferencing metadata (`xmin/ymin/xmax/ymax`, `cell_size`, `nx/ny`)

### XYZ

```bash
python tools/training/test.py \
  configs/xyz.py \
  work_dirs/xyz/latest.pth \
  --cfg-options model.test_cfg.output_dir=work_dirs/xyz/eval

python tools/evaluation/final_eval.py work_dirs/xyz/eval
```

### XYZRGB

```bash
python tools/training/test.py \
  configs/xyzrgb.py \
  work_dirs/xyzrgb/latest.pth \
  --cfg-options model.test_cfg.output_dir=work_dirs/xyzrgb/eval

python tools/evaluation/final_eval.py work_dirs/xyzrgb/eval
```

### Pretrained checkpoint (in repo)

```bash
python tools/training/test.py \
  configs/pretrained.py \
  work_dirs/pretrained/epoch_3000_fix.pth \
  --cfg-options model.test_cfg.output_dir=work_dirs/pretrained/eval

python tools/evaluation/final_eval.py work_dirs/pretrained/eval
```

## 5) TensorBoard (all runs)

```bash
nohup tensorboard --logdir work_dirs --host 0.0.0.0 --port 6006 \
  > work_dirs/tensorboard.log 2>&1 &
echo $! > work_dirs/tensorboard.pid
```

Open `http://localhost:6006` (or `<server-ip>:6006`).

Stop:

```bash
kill "$(cat work_dirs/tensorboard.pid)"
```

## 6) Current speed-oriented defaults (important)

`configs/xyz.py`:
- AMP enabled (`AmpOptimWrapper`)
- `batch_size=2`
- `num_workers=16`, `prefetch_factor=4`
- `PointSample_.num_points=400000`
- `num_queries_1dataset=128`

`configs/xyzrgb.py`:
- AMP enabled (`AmpOptimWrapper`)
- `batch_size=1`
- `num_workers=16`, `prefetch_factor=4`
- `PointSample_.num_points=400000`
- `num_queries_1dataset=128`

Also note: resume logic in `tools/training/train.py` includes a manual checkpoint load path for robust resume in this repo.
