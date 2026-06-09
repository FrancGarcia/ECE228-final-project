# BEV Fusion Inference Scripts

Command-line scripts that run the BEV fusion pipeline (YOLOv8 detection, Depth Anything V2 metric depth, and bird's-eye-view projection) on images, KITTI sequences, or video.

## Folder layout

```
bev_inference/
  inputs/            Place your own images here (.png / .jpg)
  kitti_inputs/      Populated by download_kitti.py (sequential frames)
  vid_input/         Place a landscape video here (.mp4 / .mov / ...)
  vid_frames/        Auto-extracted video frames (reset each run)
  outputs/           Cleared and rewritten on every run
  download_kitti.py  Download sequential KITTI raw drive frames
  run_inputs.py      Run pipeline on images in inputs/
  run_kitti.py       Run pipeline on KITTI images (single or batch)
  run_video.py       Run pipeline on a video (auto frame extraction)
  requirements.txt   Pinned dependencies
  README.md
```

## Setup

### Local / any machine

```bash
cd bev_inference
pip install -r requirements.txt
```

### Google Colab

Each script supports `--install-deps`, which runs the pip installs inline so you can execute directly in a Colab cell without a separate install step:

```bash
!python run_kitti.py --install-deps
```

## Quick start

### 1. Run on your own images

Place `.png` or `.jpg` images in `inputs/`, then:

```bash
python run_inputs.py
```

### 2. Run on a KITTI sequence

`download_kitti.py` fetches consecutive frames from a single KITTI raw drive. These are frames from one continuous recording, not random unrelated images.

```bash
# Download 50 sequential frames from drive 0014 (~800 MB)
python download_kitti.py

# Batch: process all downloaded frames
python run_kitti.py

# Single frame by filename stem
python run_kitti.py --image 0000000000
```

### 3. Switch to a different KITTI sequence

Specifying a different `--drive` automatically replaces the existing `kitti_inputs/` folder, so you do not need `--force`:

```bash
# Switch from drive 0014 to drive 0001 (108 frames, ~190 MB)
python download_kitti.py --drive 0001 --num 30

# List all 44 available drives with their frame counts
python download_kitti.py --list
```

The script records which drive is currently downloaded. Running the same drive again with enough frames already present skips the download.

### 4. Run on a video

Drop a landscape video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, and similar) into `vid_input/`, then:

```bash
# Auto-pick the video in vid_input/, sample at 10 fps, cap at 120 frames
python run_video.py

# Explicit file
python run_video.py --video drive.mp4

# Sample fewer frames per second (faster, fewer GIF frames)
python run_video.py --target-fps 5

# Process the whole clip (no cap)
python run_video.py --max-frames 0
```

`run_video.py` splits the video into temporally ordered frames, runs the same detection, depth, and BEV pipeline as `run_kitti.py`, and produces the same per-frame PNGs, GIFs, and validation statistics.

Frame extraction: frames are sampled at `--target-fps` (default 10 Hz, matching KITTI) rather than the video's native rate, so a 30 fps clip yields about 10 frames per second. Each frame wider than `--proc-width` (default 1280 px) is downscaled. This keeps inference fast because YOLOv8 runs at 640 px internally and Depth Anything at a fixed resolution, so larger frames cost more without improving accuracy.

Camera intrinsics: a video has no calibration file, so intrinsics are estimated from the frame size and an assumed horizontal field of view (`--hfov`, default 60 degrees, typical for phone and dashcam cameras). For metrically accurate BEV depth, pass your real values with `--fx --fy --cx --cy`.

## Output structure

Every run deletes and recreates the `outputs/` folder. For each processed frame `<stem>`, the following files are written.

### Per-frame PNGs

| File | Description |
|---|---|
| `<stem>_bev.png` | 3-panel composite (RGB detections, depth map, BEV) |
| `<stem>_rgb_det.png` | RGB image with bounding boxes and depth labels |
| `<stem>_depth.png` | Colourised metric depth map |
| `<stem>_bev_only.png` | Bird's-eye-view canvas only |

### Animated GIFs

| File | Description |
|---|---|
| `detection.gif` | All RGB and detection frames |
| `depth.gif` | All colourised depth maps |
| `bev.gif` | All BEV canvases |
| `composite.gif` | Side-by-side stitched detection, depth, and BEV |

Because the KITTI inputs are temporally sequential, the GIFs show smooth motion across the driving scene.

### Other

| File | Description |
|---|---|
| `summary.json` | Per-frame detection counts, timings, and 3D coordinates |

### Terminal output

Each run prints validation statistics to the terminal:

- BEV Fusion Summary: frame count, total detections, FPS, per-class breakdown (count, avg/min depth, avg confidence)
- Detection Depth Distribution: histogram across depth bands (0-10 m, 10-20 m, and so on)
- Confidence Statistics: mean, median, min, max confidence and counts above the 0.50 and 0.70 thresholds

## CLI reference

### `download_kitti.py`

| Flag | Default | Description |
|---|---|---|
| `--drive ID` | `0014` | KITTI raw drive ID. Run `--list` to see all 44 options |
| `--num N` | `50` | Number of consecutive frames to keep |
| `--force` | off | Re-download even if the same drive is already cached |
| `--list` | (n/a) | Print all available drives with frame counts and exit |

Specifying a different `--drive` than what is currently cached triggers a replacement download. `--force` is only needed to re-fetch the same drive.

### `run_kitti.py`

| Flag | Default | Description |
|---|---|---|
| `--image STEM` | (all) | Process a single frame by filename stem (e.g. `0000000005`) |
| `--weights PATH` | `../yolov8-finetuning/best.pt` | YOLOv8 checkpoint path |
| `--install-deps` | off | pip-install dependencies before running |

### `run_inputs.py`

| Flag | Default | Description |
|---|---|---|
| `--weights PATH` | `../yolov8-finetuning/best.pt` | YOLOv8 checkpoint path |
| `--install-deps` | off | pip-install dependencies before running |

### `run_video.py`

| Flag | Default | Description |
|---|---|---|
| `--video NAME` | (first found) | Video filename in `vid_input/` |
| `--target-fps F` | `10` | Frames sampled per second from the video |
| `--max-frames N` | `120` | Cap on processed frames (`0` = all) |
| `--proc-width W` | `1280` | Downscale frames wider than this (px) |
| `--hfov DEG` | `60` | Assumed horizontal FoV for intrinsics estimation |
| `--fx / --fy / --cx / --cy` | (estimated) | Real camera intrinsics, override `--hfov` |
| `--weights PATH` | `../yolov8-finetuning/best.pt` | YOLOv8 checkpoint path |
| `--install-deps` | off | pip-install dependencies before running |

The weights path can also be set with the `BEV_WEIGHTS` environment variable.

## Hardware compatibility

The scripts auto-detect the best available device (`cuda`, then `mps`, then `cpu`). No GPU is required; CPU inference works but is slower. Tested on:

- Google Colab (T4 GPU)
- macOS with Apple Silicon (MPS)
- Linux / Windows with NVIDIA GPU
- CPU-only machines
