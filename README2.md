# README2

## Layout

# Repo Layout (Minimal)

## data/
- splits/              # train/val/test lists
- raw/
  - las/               # LAS/LAZ inputs (incl. segmented_trees.copc.laz)
  - vectors/           # crown vectors (shp/dbf/shx/cpg)
  - plys/
    - train_val/       # training PLYs
    - test/            # test PLYs (incl. segmented_trees.ply)
- processed/
  - points/            # XYZ binary features
  - instance_mask/     # instance masks
  - semantic_mask/     # semantic masks
  - points_rgb/        # XYZ+RGB+I features (optional)
  - instance_mask_rgb/
  - semantic_mask_rgb/
- derived/
  - forainetv2_instance_data/      # XYZ processed arrays (.npy)
  - forainetv2_instance_data_rgb/  # XYZ+RGB+I processed arrays (.npy)
  - infos/                         # *_oneformer3d_infos_*.pkl
- models/             # checkpoints
- archives/           # large zips/backups

## tools/
- train.py, test.py, final_eval.py
- create_data_forainetv2*.py
- datasets/prepare.py  # unified data prep entrypoint

## configs/
- ff3d_pretrained_xyz_seminst.py
- ff3d_inst_only_xyz.py
- ff3d_inst_only_xyz_rgb_i.py

## oneformer3d/
- Core model, criterion, transforms

## work_dirs/
- pretrained checkpoint and outputs

---

## Runbook

# ForestFormer3D Minimal Runbook (GT Comparison)

This repo has been pruned to support only three workflows:
1) Evaluate pretrained model on GT
2) Train on GT (XYZ)
3) Train on GT (XYZ + RGB + intensity)

This runbook assumes you are running on a CUDA Linux GPU machine (e.g., RunPod spot).

---

## 0) Layout and Key Files

- configs/ff3d_pretrained_xyz_seminst.py
  - Pretrained baseline config (XYZ, semantic+instance)
- configs/ff3d_inst_only_xyz.py
  - Instance-only training config (XYZ)
- configs/ff3d_inst_only_xyz_rgb_i.py
  - Instance-only training config (XYZ + RGB + intensity)

- tools/datasets/las_to_ply.py
  - Convert LAS -> PLY (XYZ) with optional RGB + intensity flags

- tools/datasets/batch_load_ForAINetV2_data.py
  - Creates data/derived/forainetv2_instance_data (XYZ)
- tools/datasets/batch_load_ForAINetV2_data_rgb.py
  - Creates data/derived/forainetv2_instance_data_rgb (XYZ+RGB+I)

- tools/create_data_forainetv2.py
  - Creates info PKLs for XYZ
- tools/create_data_forainetv2_rgb.py
  - Creates info PKLs for RGB+I

- tools/test.py
  - Runs pretrained inference
- tools/train.py
  - Runs training
- tools/final_eval.py
  - Standalone evaluation (IoU/F1/PQ) from PLY outputs

- oneformer3d/
  - Core model, loss, metrics, dataset

---

## 1) Dataset Prep (XYZ)

### 1.1 Convert LAS -> PLY

From the workspace root (or inside /workspace/forestformer3d):

```
python tools/datasets/las_to_ply.py \
  /workspace/segmented_trees.las \
  /workspace/forestformer3d/data/raw/plys/test/segmented_trees.ply \
  --semantic 0
```

### 1.2 Add scan to test list

Append `segmented_trees` to:

```
data/splits/test_list.txt
```

### 1.3 Build instance data (XYZ)

```
python tools/datasets/batch_load_ForAINetV2_data.py \
  --test_scan_names_file data/splits/test_list.txt

python tools/create_data_forainetv2.py forainetv2
```

---

## 2) Dataset Prep (XYZ + RGB + intensity)

### 2.1 Convert LAS -> PLY with RGB + intensity

```
python tools/datasets/las_to_ply.py \
  /workspace/segmented_trees.las \
  /workspace/forestformer3d/data/raw/plys/test/segmented_trees.ply \
  --semantic 0 --include-rgb --include-intensity --normalize-intensity
```

### 2.2 Build instance data (RGB+I)

```
python tools/datasets/batch_load_ForAINetV2_data_rgb.py \
  --test_scan_names_file data/splits/test_list.txt

python tools/create_data_forainetv2_rgb.py forainetv2_rgb
```

---

## 3) Pretrained Inference (XYZ)

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/ff3d_pretrained_xyz_seminst.py \
  data/models/epoch_3000_fix.pth
```

Results are written under `work_dirs/...` and PLY outputs are produced by the model.

---

## 4) Train on GT (XYZ)

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/ff3d_inst_only_xyz.py \
  --work-dir work_dirs/ff3d_xyz
```

---

## 5) Train on GT (XYZ + RGB + intensity)

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/ff3d_inst_only_xyz_rgb_i.py \
  --work-dir work_dirs/ff3d_rgb
```

Note: The pretrained checkpoint is for XYZ only. RGB+I must be trained or finetuned from scratch or via a custom init.

---

## 6) Evaluate (IoU / F1 / PQ / etc.)

The eval script expects a directory of predicted .ply files.

```
python tools/final_eval.py /path/to/ply/output_dir
```

---

## 7) Multi‑GPU / Cluster Notes

- Use `tools/dist_train.sh` for distributed training if needed.
- Always run with a persistent volume (spot instances can terminate).

---

## Why Remaining

# Remaining Code: What and Why

This repo was pruned to support three workflows:
1) Pretrained inference vs GT
2) Training on GT (XYZ)
3) Training on GT (XYZ + RGB + intensity)

This file documents what remains and why it is required.

---

## configs/
- ff3d_pretrained_xyz_seminst.py
  - Pretrained baseline config (XYZ, semantic+instance).
- ff3d_inst_only_xyz.py
  - Instance-only XYZ training config (no semantic head/loss).
- ff3d_inst_only_xyz_rgb_i.py
  - Instance-only XYZ+RGB+I training config.

## data/
- splits/*.txt
  - Train/val/test lists for dataset indexing.
- data/raw/plys/train_val/*.ply, data/raw/plys/test/*.ply
  - Source PLY data (including segmented_trees.ply).

## tools/datasets/
- las_to_ply.py
  - Convert LAS to ForAINetV2 PLY with required fields (optional RGB/intensity).
- load_forainetv2_data.py, load_forainetv2_data_rgb.py
  - Parse PLY and emit feature arrays for XYZ or XYZ+RGB+I.
- batch_load_ForAINetV2_data.py, batch_load_ForAINetV2_data_rgb.py
  - Batch conversion into *_vert.npy, labels, bboxes.
- splits/*.txt
  - Train/val/test lists for dataset indexing.
- data/raw/plys/train_val/*.ply, data/raw/plys/test/*.ply
  - Source PLY data (including segmented_trees.ply).

## tools/
- train.py
  - Train XYZ or RGB+I models.
- test.py
  - Run pretrained model inference.
- final_eval.py
  - IoU/F1/PQ/MUCov evaluation on PLY outputs.
- create_data_forainetv2.py, create_data_forainetv2_rgb.py
  - Create info PKLs for XYZ and RGB+I datasets.
- converter_forainetv2.py, converter_forainetv2_rgb.py
  - Build dataset info files used by the dataloader.
- forainetv2_data_utils.py, forainetv2_data_utils_rgb.py
  - Writes feature tensors to .bin and masks to .bin for both variants.
- base_modules.py
  - Used by oneformer3d/oneformer3d*.py (MLP/Seq helpers).

## oneformer3d/
Core model + loss + data pipeline. Still required even for instance-only objectives:
- oneformer3d.py, oneformer3d_speedup_v1.py
  - Model implementation (speedup_v1 is used by configs).
- spconv_unet.py, query_decoder.py
  - Backbone + decoder.
- transforms_3d.py, formatting.py, loading.py, data_preprocessor.py
  - Dataset transforms and packing.
- unified_criterion.py, semantic_criterion.py, instance_criterion.py, panoptic_losses.py
  - Losses and matching. Semantic loss is still used by the current model.
- unified_metric.py
  - Evaluation metric used by configs.
- structures.py, forainetv2_dataset.py
  - Dataset plumbing.

## work_dirs/
- data/models/epoch_3000_fix.pth
  - Pretrained checkpoint for XYZ inference.

---

## Not removed (by design)
- Semantic loss and semantic head code
  - The current model architecture couples semantic and instance heads.
  - Removing semantic parts would require a deeper refactor to instance‑only.

If you want an instance‑only refactor, say so and I will remove the semantic head/loss cleanly.
