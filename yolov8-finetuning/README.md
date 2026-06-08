# YOLOv8 Fine-Tuning — KITTI Object Detection

Stage 1 of the pipeline: 2D object detection.

Fine-tunes **YOLOv8m** (from COCO-pretrained weights) on the KITTI object-detection dataset, filtered to three classes relevant for driving: **Car, Pedestrian, Cyclist**.

## Contents

| File | Description |
|---|---|
| `yolov8_kitti_finetune.ipynb` | End-to-end training notebook: dataset prep → fine-tuning → mAP evaluation → inference |
| `best.pt` | Trained YOLOv8m checkpoint (~50 MB). Used as the default weights by every script in `bev_inference/`. |

## Notebook workflow

1. Pin `numpy==1.26.4` and install `ultralytics`
2. Download KITTI, filter to 3 classes, write the dataset YAML
3. Fine-tune YOLOv8m from COCO-pretrained weights
4. Evaluate — mAP@50, mAP@50:95, per-class AP on the validation set
5. Plot training curves (loss, mAP, precision/recall)
6. Run inference on sample validation images
7. Export `best.pt`

## Requirements

Google Colab with a GPU (T4 or better): **Runtime → Change runtime type → GPU**.

The exported `best.pt` is consumed downstream by `bev_fusion/`, `bev_trajectory_modelling/`, and the `bev_inference/` scripts.
