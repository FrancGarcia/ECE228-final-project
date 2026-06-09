# BEV Fusion (Scene Reconstruction)

Stage 3 of the pipeline: fusing detection and depth into a bird's-eye-view.

Combines YOLOv8m object detections with Depth Anything V2 metric depth to produce top-down bird's-eye-view (BEV) representations of street scenes.

## Contents

| File | Description |
|---|---|
| `bev_fusion_1.ipynb` | The BEV fusion notebook (detection and depth, 3D unprojection, BEV render) |
| `bev_outputs/` | Sample rendered outputs (RGB with detections, depth, BEV, composite) |
| `bev_outputs.zip` | Zipped copy of the sample outputs |

## Per-frame pipeline

1. Run YOLOv8m to get bounding boxes `(x1, y1, x2, y2)` and class ids
2. Run Depth Anything V2 to get a dense depth map in metres
3. For each detection, sample depth at the bottom-centre of the box (approximates the ground-contact point)
4. Use the KITTI camera intrinsics `K` to unproject the pixel to 3D `(X, Y, Z)`
5. Project the 3D points onto a top-down BEV canvas (X horizontal, Z forward)
6. Render side by side: RGB with detections, depth map, BEV grid

## Requirements

Google Colab with a GPU (T4 or better).

The logic in this notebook is packaged into command-line scripts in [`../bev_inference/`](../bev_inference/), which run the same fusion on custom images, KITTI sequences, or video and export GIFs and validation statistics.
