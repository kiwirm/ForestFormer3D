# ForestFormer3D Cass Runbook (Instance-Only)

This runbook documents the Cass workflow with RGB+I data prep and instance-only training.
Assumes a CUDA Linux GPU machine and that `.venv` is set up.

---

## Repo Layout (Current)

## data/
- splits/
  - original/             # original_*_list.txt
  - combined/             # combined_*_list.txt
  - cass/                 # cass_*_list.txt
- raw/
  - las/                  # raw LAS/LAZ inputs
  - vectors/              # crown vectors (shp/dbf/shx/cpg)
- intermediate/
  - cass_labeled.las      # labeled LAS after polygon join
- labeled/
  - plys/
    - train_val/
      - cass/             # Cass train/val tiles
      - other/            # all non-Cass labeled PLYs
    - test/
      - cass/             # Cass test tiles
      - other/            # all non-Cass labeled PLYs
- processed/
  - points/           # XYZ+RGB+I .bin features
  - instance_mask/    # instance masks .bin
  - semantic_mask/    # semantic masks .bin
- derived/
  - instance_data/  # cached arrays (.npy)
  - infos/                         # *_oneformer3d_infos_*.pkl
- models/                 # checkpoints
- archives/               # large zips/backups

## tools/
- run_cass_pipeline.sh    # end-to-end Cass pipeline
- datasets/
  - las_to_ply.py
  - las_to_ply_tiles.py
  - parse_ply.py
  - preprocess_dataset.py
  - assign_tree_id_from_shp.py
- prep/
  - build_infos.py
  - build_info_index.py
  - cache_builder.py
  - patch_infos.py
- training/
  - train.py
  - test.py
- evaluation/
  - final_eval.py
- support/
  - base_modules.py
  - plyutils.py

## configs/
- cass_pretrained_seminst.py
- ff3d_inst_only_xyz_cass.py
- ff3d_inst_only_xyz_rgb_i_cass.py

---

## Cass Prep (End-to-End)

This pipeline expects:
- `data/raw/las/cass/cass.segment.crop.las`
- `data/raw/vectors/tree_crowns.shp`

Run:
```
source .venv/bin/activate
export PYTHONPATH=.

bash tools/run_cass_pipeline.sh
```

This produces:
- Labeled tiles in `data/labeled/plys/{train_val,test}/cass`
- Split lists in `data/splits/cass/`
- Cached arrays + bins under `data/derived/` and `data/processed/`
- PKL infos in `data/derived/infos/`

---

## Workflows

### 1) Inference Cass with existing checkpoint → eval

The checkpoint in `data/models/epoch_3000_fix.pth` is **semantic+instance** and matches:
`configs/cass_pretrained_seminst.py`

```
source .venv/bin/activate
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python tools/training/test.py \
  configs/cass_pretrained_seminst.py \
  data/models/epoch_3000_fix.pth \
  --cfg-options test_cfg.output_dir=work_dirs/cass_pretrained_infer
```

Eval:
```
python tools/evaluation/final_eval.py work_dirs/cass_pretrained_infer
```

---

### 2) Train Cass (XYZ-only) → inference → eval

The XYZ-only config ignores RGB+I via `use_dim=[0,1,2]` but still reads the RGB+I bins.

Train:
```
source .venv/bin/activate
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python tools/training/train.py \
  configs/ff3d_inst_only_xyz_cass.py \
  --work-dir work_dirs/cass_xyz
```

Inference:
```
CUDA_VISIBLE_DEVICES=0 python tools/training/test.py \
  configs/ff3d_inst_only_xyz_cass.py \
  work_dirs/cass_xyz/latest.pth \
  --cfg-options test_cfg.output_dir=work_dirs/cass_xyz_infer
```

Eval:
```
python tools/evaluation/final_eval.py work_dirs/cass_xyz_infer
```

---

### 3) Train Cass (RGB+I) → inference → eval

Train:
```
source .venv/bin/activate
export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0 python tools/training/train.py \
  configs/ff3d_inst_only_xyz_rgb_i_cass.py \
  --work-dir work_dirs/cass_rgb
```

Inference:
```
CUDA_VISIBLE_DEVICES=0 python tools/training/test.py \
  configs/ff3d_inst_only_xyz_rgb_i_cass.py \
  work_dirs/cass_rgb/latest.pth \
  --cfg-options test_cfg.output_dir=work_dirs/cass_rgb_infer
```

Eval:
```
python tools/evaluation/final_eval.py work_dirs/cass_rgb_infer
```

---

## Notes

- All data prep produces RGB+I caches. XYZ-only configs simply ignore extra dims.
- The pretrained checkpoint in `data/models` is **not** compatible with instance-only configs.
- If you need different output dirs, change `test_cfg.output_dir` in the test command.
