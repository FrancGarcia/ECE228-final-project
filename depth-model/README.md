# Depth Model (Depth Anything V2, KITTI Evaluation)

Stage 2 of the pipeline: metric depth estimation.

Loads and evaluates Depth Anything V2 (`depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`) on KITTI validation images and compares the predicted per-pixel metric depth against ground-truth LiDAR.

## Contents

| File | Description |
|---|---|
| `depth_model.ipynb` | Loads the model, runs depth prediction on KITTI samples, and compares against LiDAR depth |

## Why this model

Depth Anything V2 (Metric, Outdoor) produces dense per-pixel depth in metres from a single RGB image, with no stereo or LiDAR needed at inference. In the full pipeline, this depth is sampled at the bottom-centre of each detected bounding box to recover the object's 3D position.

## Requirements

Best run with a GPU. The model is downloaded automatically from HuggingFace on first use, so no checkpoint is committed to this folder.

This depth model is combined with the YOLOv8 detector in `bev_fusion/` and the scripts in `bev_inference/`.
