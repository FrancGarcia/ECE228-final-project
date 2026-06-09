#!/usr/bin/env python3
"""
BEV Fusion inference on a landscape video.

Takes a video file from ``vid_input/`` (any common format: mp4, mov,
avi, mkv, webm, ...), splits it into temporally ordered frames, runs
YOLOv8 detection + Depth Anything V2 metric depth, fuses them into a
bird's-eye-view, and writes all outputs to ``outputs/`` - exactly like
``run_kitti.py``, but the frames come from a video instead of disk.

Frame extraction is "chunked" by sampling at a target frame rate
(default 10 Hz, matching KITTI) and optionally down-scaling each frame
to an optimal processing width (default 1280 px). This keeps inference
fast and memory-light without sacrificing detection/depth accuracy -
YOLOv8 internally works at 640 px and Depth Anything at its own fixed
resolution, so widths beyond ~1280 px add cost with no quality gain.

Outputs include per-frame PNGs and animated GIFs (detection, depth,
BEV, composite). Validation statistics are printed to the terminal.

The ``outputs/`` folder is **deleted and recreated** on every run.

Camera intrinsics
-----------------
A monocular video has no calibration file, so intrinsics are estimated
from the (processed) frame size and an assumed horizontal field of view
(``--hfov``, default 60 deg, typical for phone / dashcam main cameras):

    fx = fy = (W / 2) / tan(HFOV / 2)
    cx = W / 2 ,  cy = H / 2

If you know your camera's real intrinsics, pass them with
``--fx --fy --cx --cy`` for metrically accurate BEV depth placement.

Modes
-----
    python run_video.py                          # auto-pick video in vid_input/
    python run_video.py --video drive.mp4        # explicit file
    python run_video.py --target-fps 5           # sample 5 frames / sec
    python run_video.py --max-frames 80          # cap total frames
    python run_video.py --hfov 70                # wider lens
    python run_video.py --fx 1000 --fy 1000 --cx 960 --cy 540
    python run_video.py --install-deps           # pip-install first
"""

import argparse
import glob as _glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Paths (relative to this script)
SCRIPT_DIR      = Path(__file__).resolve().parent
VID_INPUT_DIR   = SCRIPT_DIR / "vid_input"
VID_FRAMES_DIR  = SCRIPT_DIR / "vid_frames"
OUTPUT_DIR      = SCRIPT_DIR / "outputs"
DEFAULT_WEIGHTS = SCRIPT_DIR.parent / "yolov8-finetuning" / "best.pt"

# Camera intrinsics (set at runtime from video size + HFOV)
FX = 721.5377
FY = 721.5377
CX = 609.5593
CY = 172.8540

# Inference thresholds
YOLO_CONF   = 0.30
YOLO_IOU    = 0.45
DEPTH_MAX_M = 80.0
DEPTH_MIN_M = 1.0

# BEV canvas parameters
BEV_RANGE_Z = 50.0
BEV_RANGE_X = 30.0
BEV_PPM     = 10
BEV_H       = int(BEV_RANGE_Z * BEV_PPM)
BEV_W       = int(BEV_RANGE_X * 2 * BEV_PPM)

# Class definitions
CLASS_NAMES    = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
CLASS_BGR      = {0: (235, 99, 37), 1: (74, 163, 22), 2: (6, 119, 217)}
CLASS_RGB      = {
    0: (0.145, 0.388, 0.922),
    1: (0.086, 0.639, 0.290),
    2: (0.851, 0.467, 0.024),
}
CLASS_RADIUS_M = {0: 1.5, 1: 0.4, 2: 0.6}

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# Video / frame-extraction parameters
VIDEO_EXTS      = {".mp4", ".mov", ".avi", ".mkv", ".webm",
                   ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
DEFAULT_TGT_FPS = 10.0     # KITTI raw is 10 Hz - good default for BEV
DEFAULT_MAXFR   = 120      # cap to keep runtime sane (None = unlimited)
PROC_MAX_WIDTH  = 1280     # downscale wider frames to this for processing
DEFAULT_HFOV    = 60.0     # assumed horizontal FoV (deg) when no intrinsics

# GIF parameters
FRAME_DURATION_MS = 150
GIF_SCALE         = 0.6


# Helpers

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def install_deps() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "numpy==1.26.4", "--force-reinstall"],
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics>=8.3.225", "transformers", "accelerate",
         "opencv-python-headless", "matplotlib", "scipy", "--upgrade"],
    )
    print("Dependencies installed.")


# Video frame extraction (chunking)

def find_video(explicit: str | None) -> Path:
    if explicit:
        cand = Path(explicit)
        if not cand.is_absolute():
            cand = VID_INPUT_DIR / explicit
        if not cand.exists():
            print(f"ERROR: Video not found: {cand}")
            sys.exit(1)
        return cand

    if not VID_INPUT_DIR.exists():
        print(f"ERROR: {VID_INPUT_DIR} does not exist. Create it and add a video.")
        sys.exit(1)

    videos = sorted(p for p in VID_INPUT_DIR.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"ERROR: No video files found in {VID_INPUT_DIR}")
        print(f"       Supported: {', '.join(sorted(VIDEO_EXTS))}")
        sys.exit(1)
    if len(videos) > 1:
        print(f"Found {len(videos)} videos; using first: {videos[0].name}")
        print(f"  (use --video NAME to pick a specific one)")
    return videos[0]


def extract_frames(video_path: Path, target_fps: float,
                   max_frames: int | None, proc_width: int) -> tuple:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open {video_path}")
        print("       The codec may be unsupported by this OpenCV build.")
        sys.exit(1)

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if src_fps <= 0:
        src_fps = 30.0  # fallback for containers that don't report fps

    stride = max(1, int(round(src_fps / max(target_fps, 0.1))))

    # Resize factor: only ever downscale, never upscale
    scale = 1.0
    if src_w > proc_width:
        scale = proc_width / float(src_w)
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))

    print(f"Video: {video_path.name}")
    print(f"  Source     : {src_w}x{src_h} @ {src_fps:.2f} fps, "
          f"{total_n} frames ({total_n / src_fps:.1f}s)" if total_n else
          f"  Source     : {src_w}x{src_h} @ {src_fps:.2f} fps")
    print(f"  Sampling   : every {stride} frame(s) -> ~{src_fps / stride:.1f} fps")
    if scale < 1.0:
        print(f"  Compressing: {src_w}x{src_h} -> {out_w}x{out_h} "
              f"({scale * 100:.0f}% - optimal for inference)")
    else:
        print(f"  Resolution : {out_w}x{out_h} (no downscale needed)")

    reset_dir(VID_FRAMES_DIR)

    saved, idx, kept = [], 0, 0
    pbar = tqdm(total=(total_n // stride if total_n else None),
                desc="Extracting frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            fp = VID_FRAMES_DIR / f"frame_{kept:06d}.png"
            cv2.imwrite(str(fp), frame)
            saved.append(fp)
            kept += 1
            pbar.update(1)
            if max_frames and kept >= max_frames:
                print(f"\n  Reached --max-frames {max_frames}; stopping extraction.")
                break
        idx += 1
    pbar.close()
    cap.release()

    if not saved:
        print("ERROR: No frames could be extracted from the video.")
        sys.exit(1)

    print(f"  Extracted  : {len(saved)} frames -> {VID_FRAMES_DIR}/")
    return saved, out_w, out_h


# Depth inference / projection

def predict_depth(pil_image: Image.Image, depth_model, depth_processor, device: str) -> np.ndarray:
    with torch.inference_mode():
        inputs  = depth_processor(images=pil_image, return_tensors="pt").to(device)
        outputs = depth_model(**inputs)
        pred    = outputs.predicted_depth.squeeze().cpu().float().numpy()
    w, h = pil_image.size
    pred = np.array(Image.fromarray(pred).resize((w, h), Image.BILINEAR))
    return pred.clip(0, DEPTH_MAX_M)


def unproject(u: float, v: float, depth_m: float):
    X = (u - CX) * depth_m / FX
    Y = (v - CY) * depth_m / FY
    Z = depth_m
    return X, Y, Z


def world_to_bev(X: float, Z: float):
    bev_u = int(BEV_W / 2 + X * BEV_PPM)
    bev_v = int(BEV_H     - Z * BEV_PPM)
    if 0 <= bev_u < BEV_W and 0 <= bev_v < BEV_H:
        return bev_u, bev_v
    return None


def render_bev(detections: list) -> np.ndarray:
    canvas = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

    grid_color = (40, 40, 40)
    for z_m in range(0, int(BEV_RANGE_Z) + 1, 10):
        y_px = int(BEV_H - z_m * BEV_PPM)
        if 0 <= y_px < BEV_H:
            cv2.line(canvas, (0, y_px), (BEV_W, y_px), grid_color, 1)
            cv2.putText(canvas, f"{z_m}m", (2, max(y_px - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    for x_m in range(-int(BEV_RANGE_X), int(BEV_RANGE_X) + 1, 10):
        x_px = int(BEV_W / 2 + x_m * BEV_PPM)
        if 0 <= x_px < BEV_W:
            cv2.line(canvas, (x_px, 0), (x_px, BEV_H), grid_color, 1)

    ego_u, ego_v = BEV_W // 2, BEV_H - 1
    cv2.drawMarker(canvas, (ego_u, ego_v), (255, 255, 255),
                   cv2.MARKER_TRIANGLE_UP, 12, 2)

    for det in detections:
        cls_id = det["cls_id"]
        bev_pt = world_to_bev(det["X"], det["Z"])
        if bev_pt is None:
            continue
        bev_u, bev_v = bev_pt
        color  = CLASS_BGR.get(cls_id, (200, 200, 200))
        radius = max(1, int(CLASS_RADIUS_M.get(cls_id, 1.0) * BEV_PPM))
        cv2.circle(canvas, (bev_u, bev_v), radius, color, -1)
        cv2.circle(canvas, (bev_u, bev_v), radius, (255, 255, 255), 1)
        label = f'{CLASS_NAMES.get(cls_id, "?")[0]} {det["Z"]:.0f}m'
        cv2.putText(canvas, label, (bev_u + radius + 1, bev_v + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    return canvas


def colorise_depth(depth_map: np.ndarray, vmin: float = 0.0, vmax: float = DEPTH_MAX_M) -> np.ndarray:
    d_norm = np.clip((depth_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba   = plt.get_cmap("plasma")(d_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# Per-frame fusion

def process_image(img_path: Path, yolo_model, depth_model, depth_processor, device: str) -> dict:
    t0 = time.time()

    pil_img = Image.open(img_path).convert("RGB")
    img_bgr = cv2.imread(str(img_path))
    img_h, img_w = img_bgr.shape[:2]

    yolo_results = yolo_model(img_bgr, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    depth_map    = predict_depth(pil_img, depth_model, depth_processor, device)

    detections = []
    annotated  = img_bgr.copy()

    for box in yolo_results.boxes:
        cls_id          = int(box.cls)
        conf            = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        sample_u = int((x1 + x2) / 2)
        sample_v = min(y2, img_h - 1)

        u0, u1 = max(0, sample_u - 2), min(img_w, sample_u + 3)
        v0, v1 = max(0, sample_v - 2), min(img_h, sample_v + 3)
        depth_m = float(depth_map[v0:v1, u0:u1].mean())

        if depth_m < DEPTH_MIN_M or depth_m > DEPTH_MAX_M:
            continue

        X, Y, Z = unproject(sample_u, sample_v, depth_m)
        detections.append({"cls_id": cls_id, "conf": conf,
                           "X": X, "Y": Y, "Z": Z,
                           "box": [x1, y1, x2, y2]})

        bgr   = CLASS_BGR.get(cls_id, (200, 200, 200))
        label = f'{CLASS_NAMES.get(cls_id, "?")} {conf:.2f} | {depth_m:.1f}m'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        cv2.rectangle(annotated, (x1, y1 - 14), (x1 + len(label) * 7, y1), bgr, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    bev_canvas = render_bev(detections)
    depth_vis  = colorise_depth(depth_map)
    elapsed    = time.time() - t0

    # Composite figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 5), gridspec_kw={"wspace": 0.04})
    fig.suptitle(f"{img_path.stem}  |  {len(detections)} detections  |  {elapsed:.2f}s",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB + Detections (depth labelled)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(depth_vis)
    axes[1].set_title(f"Depth Anything V2 (0 - {DEPTH_MAX_M:.0f}m)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(bev_canvas, cv2.COLOR_BGR2RGB), origin="upper",
                   extent=[-BEV_RANGE_X, BEV_RANGE_X, 0, BEV_RANGE_Z])
    axes[2].set_title("Bird's-Eye View (top-down)", fontsize=10)
    axes[2].set_xlabel("Lateral X (m)")
    axes[2].set_ylabel("Forward Z (m)")
    axes[2].set_xlim(-BEV_RANGE_X, BEV_RANGE_X)
    axes[2].set_ylim(0, BEV_RANGE_Z)
    legend_patches = [mpatches.Patch(color=CLASS_RGB[i], label=CLASS_NAMES[i])
                      for i in sorted(CLASS_NAMES)]
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.7)

    composite_path = OUTPUT_DIR / f"{img_path.stem}_bev.png"
    plt.savefig(str(composite_path), bbox_inches="tight", dpi=130)
    plt.close(fig)

    cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_rgb_det.png"), annotated)
    cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_depth.png"),
                cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_bev_only.png"), bev_canvas)

    return {"frame": img_path.stem, "n_det": len(detections),
            "elapsed_s": round(elapsed, 3), "detections": detections}


# GIF generation

def _load_frames_sorted(pattern: str, scale: float = 1.0) -> list:
    paths = sorted(_glob.glob(pattern))
    frames = []
    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            h, w = img_rgb.shape[:2]
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(img_rgb))
    return frames


def _save_gif(frames: list, out_path: str, duration_ms: int = 150) -> bool:
    if not frames:
        print(f"  No frames for {os.path.basename(out_path)} - skipped.")
        return False
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {os.path.basename(out_path):<35} "
          f"{len(frames)} frames  {size_mb:.1f} MB")
    return True


def generate_gifs() -> None:
    out = str(OUTPUT_DIR)
    print("\nGenerating GIF animations ...")
    print(f"  Scale: {GIF_SCALE}x  |  Frame duration: {FRAME_DURATION_MS}ms")

    _save_gif(_load_frames_sorted(f"{out}/*_rgb_det.png", GIF_SCALE),
              f"{out}/detection.gif", FRAME_DURATION_MS)

    _save_gif(_load_frames_sorted(f"{out}/*_depth.png", GIF_SCALE),
              f"{out}/depth.gif", FRAME_DURATION_MS)

    _save_gif(_load_frames_sorted(f"{out}/*_bev_only.png", GIF_SCALE),
              f"{out}/bev.gif", FRAME_DURATION_MS)

    # Composite: stitch detection | depth | BEV per frame
    det_paths = sorted(_glob.glob(f"{out}/*_rgb_det.png"))
    dep_paths = sorted(_glob.glob(f"{out}/*_depth.png"))
    bev_paths = sorted(_glob.glob(f"{out}/*_bev_only.png"))

    composite_pil = []
    n = min(len(det_paths), len(dep_paths), len(bev_paths))
    for i in range(n):
        det_bgr = cv2.imread(det_paths[i])
        dep_bgr = cv2.imread(dep_paths[i])
        bev_bgr = cv2.imread(bev_paths[i])
        if det_bgr is None or dep_bgr is None or bev_bgr is None:
            continue

        target_h = det_bgr.shape[0]

        def _rh(img, h):
            oh, ow = img.shape[:2]
            return cv2.resize(img, (int(ow * h / oh), h), interpolation=cv2.INTER_AREA)

        det_r = _rh(det_bgr, target_h)
        dep_r = _rh(dep_bgr, target_h)
        bev_r = _rh(bev_bgr, target_h)

        sep = np.ones((target_h, 3, 3), dtype=np.uint8) * 200
        row = np.concatenate([det_r, sep, dep_r, sep, bev_r], axis=1)

        sh = int(row.shape[0] * GIF_SCALE)
        sw = int(row.shape[1] * GIF_SCALE)
        row = cv2.resize(row, (sw, sh), interpolation=cv2.INTER_AREA)
        composite_pil.append(Image.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)))

    _save_gif(composite_pil, f"{out}/composite.gif", FRAME_DURATION_MS)
    print("GIF generation complete.")


# Validation statistics

def print_validation_stats(frame_stats: list) -> None:
    if not frame_stats:
        return

    total_dets = sum(s["n_det"] for s in frame_stats)
    avg_time   = np.mean([s["elapsed_s"] for s in frame_stats])

    sep = "=" * 65
    print(f"\n{sep}")
    print("BEV FUSION SUMMARY")
    print(sep)
    print(f"  Frames processed     : {len(frame_stats)}")
    print(f"  Total detections     : {total_dets}")
    print(f"  Avg detections/frame : {total_dets / max(len(frame_stats), 1):.1f}")
    print(f"  Avg time/frame       : {avg_time:.2f}s")
    if avg_time > 0:
        print(f"  Effective FPS        : {1 / avg_time:.2f}")
    print(sep)

    # Per-class breakdown
    class_counts   = {name: 0 for name in CLASS_NAMES.values()}
    depth_by_class = {name: [] for name in CLASS_NAMES.values()}
    conf_by_class  = {name: [] for name in CLASS_NAMES.values()}
    for s in frame_stats:
        for det in s["detections"]:
            name = CLASS_NAMES.get(det["cls_id"], "Unknown")
            class_counts[name] += 1
            depth_by_class[name].append(det["Z"])
            conf_by_class[name].append(det["conf"])

    print(f'  {"Class":<14} {"Count":>7} {"Avg depth(m)":>14} '
          f'{"Min depth(m)":>14} {"Avg conf":>10}')
    print(f"  {'-' * 63}")
    for name in CLASS_NAMES.values():
        depths = depth_by_class[name]
        confs  = conf_by_class[name]
        avg_d  = np.mean(depths) if depths else 0.0
        min_d  = np.min(depths)  if depths else 0.0
        avg_c  = np.mean(confs)  if confs  else 0.0
        print(f"  {name:<14} {class_counts[name]:>7} {avg_d:>14.1f} "
              f"{min_d:>14.1f} {avg_c:>10.3f}")
    print(sep)

    # Detection depth distribution
    depth_bands = {"0-10m": 0, "10-20m": 0, "20-30m": 0, "30-40m": 0, ">40m": 0}
    for s in frame_stats:
        for det in s["detections"]:
            z = det["Z"]
            if   z < 10: depth_bands["0-10m"]  += 1
            elif z < 20: depth_bands["10-20m"] += 1
            elif z < 30: depth_bands["20-30m"] += 1
            elif z < 40: depth_bands["30-40m"] += 1
            else:        depth_bands[">40m"]   += 1

    total_d = sum(depth_bands.values()) or 1
    print("\nDETECTION DEPTH DISTRIBUTION")
    print("-" * 50)
    for band, cnt in depth_bands.items():
        bar = "#" * int(30 * cnt / total_d)
        print(f"  {band:>8}: {cnt:4d}  ({100 * cnt / total_d:5.1f}%)  {bar}")

    # Confidence distribution
    all_confs = [det["conf"] for s in frame_stats for det in s["detections"]]
    if all_confs:
        print(f"\nCONFIDENCE STATISTICS")
        print("-" * 50)
        print(f"  Mean confidence      : {np.mean(all_confs):.3f}")
        print(f"  Median confidence    : {np.median(all_confs):.3f}")
        print(f"  Min confidence       : {np.min(all_confs):.3f}")
        print(f"  Max confidence       : {np.max(all_confs):.3f}")
        print(f"  Conf >= 0.50         : {sum(c >= 0.50 for c in all_confs)}/{len(all_confs)}")
        print(f"  Conf >= 0.70         : {sum(c >= 0.70 for c in all_confs)}/{len(all_confs)}")

    # Frames with zero detections
    zero_frames = sum(1 for s in frame_stats if s["n_det"] == 0)
    if zero_frames:
        print(f"\n  Frames with 0 detections: {zero_frames}/{len(frame_stats)}")

    print(sep)


def save_summary(frame_stats: list, meta: dict) -> None:
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"meta": meta, "frames": frame_stats}, f, indent=2, default=str)
    print(f"  Summary written to {summary_path}")


# Intrinsics

def set_intrinsics(width: int, height: int, args) -> dict:
    global FX, FY, CX, CY

    if args.fx and args.fy and args.cx is not None and args.cy is not None:
        FX, FY, CX, CY = args.fx, args.fy, args.cx, args.cy
        source = "user-supplied"
    else:
        hfov_rad = math.radians(args.hfov)
        FX = (width / 2.0) / math.tan(hfov_rad / 2.0)
        FY = FX                       # assume square pixels
        CX = width / 2.0
        CY = height / 2.0
        source = f"estimated (HFOV={args.hfov:.0f}\u00b0)"

    print(f"Camera intrinsics ({source}):")
    print(f"  fx={FX:.1f}  fy={FY:.1f}  cx={CX:.1f}  cy={CY:.1f}")
    return {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "source": source}


# Main

def main():
    parser = argparse.ArgumentParser(
        description="BEV Fusion - run on a landscape video from vid_input/")
    parser.add_argument("--video", type=str, default=None,
                        help="Video filename in vid_input/ (default: first found)")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TGT_FPS,
                        help=f"Frames sampled per second (default {DEFAULT_TGT_FPS})")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAXFR,
                        help=f"Max frames to process (default {DEFAULT_MAXFR}; 0 = all)")
    parser.add_argument("--proc-width", type=int, default=PROC_MAX_WIDTH,
                        help=f"Downscale frames wider than this (default {PROC_MAX_WIDTH})")
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV,
                        help=f"Assumed horizontal FoV in deg (default {DEFAULT_HFOV})")
    parser.add_argument("--fx", type=float, default=None, help="Camera fx (overrides --hfov)")
    parser.add_argument("--fy", type=float, default=None, help="Camera fy (overrides --hfov)")
    parser.add_argument("--cx", type=float, default=None, help="Camera cx (overrides --hfov)")
    parser.add_argument("--cy", type=float, default=None, help="Camera cy (overrides --hfov)")
    parser.add_argument("--weights", type=str,
                        default=os.environ.get("BEV_WEIGHTS", str(DEFAULT_WEIGHTS)),
                        help="Path to YOLOv8 checkpoint (default: ../yolov8-finetuning/best.pt)")
    parser.add_argument("--install-deps", action="store_true",
                        help="pip-install required packages before running")
    args = parser.parse_args()

    if args.install_deps:
        install_deps()

    max_frames = None if args.max_frames in (0, None) else args.max_frames

    video_path = find_video(args.video)
    frame_paths, proc_w, proc_h = extract_frames(
        video_path, args.target_fps, max_frames, args.proc_width)

    intr = set_intrinsics(proc_w, proc_h, args)

    reset_dir(OUTPUT_DIR)

    device = get_device()
    print(f"Device: {device}")

    from ultralytics import YOLO
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: YOLO weights not found at {weights}")
        sys.exit(1)

    print(f"Loading YOLO checkpoint: {weights}")
    yolo_model = YOLO(str(weights))

    print(f"Loading depth model: {DEPTH_MODEL_ID}")
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    depth_model     = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
    depth_model.eval()
    print("Both models ready.\n")

    frame_stats = []
    for img_path in tqdm(frame_paths, desc="BEV fusion"):
        stats = process_image(img_path, yolo_model, depth_model, depth_processor, device)
        frame_stats.append(stats)

    meta = {"video": video_path.name, "frames": len(frame_stats),
            "proc_size": [proc_w, proc_h], "target_fps": args.target_fps,
            "intrinsics": intr}

    print_validation_stats(frame_stats)
    save_summary(frame_stats, meta)
    generate_gifs()
    print(f"\nOutputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
