#!/usr/bin/env python3
"""
Download a KITTI raw driving sequence for BEV Fusion inference.

Downloads a full KITTI raw drive (sequential frames from a single
camera), extracts the left-colour camera images (``image_02/data/``),
and copies the first ``--num`` (default 50) **consecutive** frames
into ``kitti_inputs/``.

This produces temporally coherent data suitable for GIF animations,
multi-object tracking, and kinematic analysis.

Available drives (use ``--drive``)
----------------------------------
    0001   108 frames  |  0002   77 frames  |  0005  154 frames
    0009   447 frames  |  0011   233 frames  |  0013  144 frames
    0014   314 frames  (default)
    0015   297 frames  |  0017   114 frames  |  0018   270 frames
    0019   400 frames  |  0020   86 frames   |  0022   800 frames
    0023   474 frames  |  0027   188 frames  |  0028   429 frames
    0029   400 frames  |  0032   390 frames  |  0035   131 frames
    0036   801 frames  |  0039   395 frames  |  0046   125 frames
    0048   22 frames   |  0051   438 frames  |  0052   78 frames
    0056   294 frames  |  0057   361 frames  |  0059   373 frames
    0060   78 frames   |  0061   694 frames  |  0064   570 frames
    0070   420 frames  |  0079   100 frames  |  0084   383 frames
    0086   706 frames  |  0091   339 frames  |  0093   433 frames
    0095   268 frames  |  0096   475 frames  |  0101   936 frames
    0104   312 frames  |  0106   174 frames  |  0113   80 frames
    0117   660 frames

Usage
-----
    python download_kitti.py                   # 50 frames from drive 0014
    python download_kitti.py --num 30          # first 30 frames
    python download_kitti.py --drive 0001      # use a different drive
    python download_kitti.py --force           # re-download even if present
    python download_kitti.py --list            # list all available drives
"""

import argparse
import glob as _glob
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

SCRIPT_DIR  = Path(__file__).resolve().parent
KITTI_DIR   = SCRIPT_DIR / "kitti_inputs"
RAW_DATE    = "2011_09_26"
DEFAULT_NUM = 50

DRIVE_URL_TEMPLATE = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    "raw_data/{date}_drive_{drive}/{date}_drive_{drive}_sync.zip"
)

KNOWN_DRIVES = {
    "0001": 108,  "0002":  77,  "0005": 154,  "0009": 447,
    "0011": 233,  "0013": 144,  "0014": 314,  "0015": 297,
    "0017": 114,  "0018": 270,  "0019": 400,  "0020":  86,
    "0022": 800,  "0023": 474,  "0027": 188,  "0028": 429,
    "0029": 400,  "0032": 390,  "0035": 131,  "0036": 801,
    "0039": 395,  "0046": 125,  "0048":  22,  "0051": 438,
    "0052":  78,  "0056": 294,  "0057": 361,  "0059": 373,
    "0060":  78,  "0061": 694,  "0064": 570,  "0070": 420,
    "0079": 100,  "0084": 383,  "0086": 706,  "0091": 339,
    "0093": 433,  "0095": 268,  "0096": 475,  "0101": 936,
    "0104": 312,  "0106": 174,  "0113":  80,  "0117": 660,
}


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb  = downloaded / 1e6
        tot = total_size / 1e6
        print(f"\r  Downloading: {mb:.1f} / {tot:.1f} MB ({pct:.0f}%)", end="", flush=True)
    else:
        print(f"\r  Downloaded: {downloaded / 1e6:.1f} MB", end="", flush=True)


def _print_available_drives():
    print("Available KITTI raw drives (2011_09_26):\n")
    items = sorted(KNOWN_DRIVES.items())
    for i in range(0, len(items), 4):
        row = items[i:i+4]
        parts = [f"  {did}  ({frames:>4} frames)" for did, frames in row]
        print("  ".join(parts))
    print(f"\nTotal: {len(items)} drives")


def _read_meta() -> dict:
    meta_file = KITTI_DIR / ".kitti_meta"
    if meta_file.exists():
        try:
            import json
            return json.loads(meta_file.read_text())
        except Exception:
            pass
    return {}


def _write_meta(drive: str, num: int) -> None:
    import json
    meta_file = KITTI_DIR / ".kitti_meta"
    meta_file.write_text(json.dumps({"drive": drive, "num": num}))


def download_and_extract(drive: str, num: int, force: bool) -> None:
    existing = sorted(KITTI_DIR.glob("*.png"))
    meta = _read_meta()
    current_drive = meta.get("drive")

    if existing and current_drive and current_drive != drive:
        print(f"Switching drive: {current_drive} → {drive}  (replacing {len(existing)} existing frames)")
        force = True

    if len(existing) >= num and not force:
        print(f"Already have {len(existing)} sequential frames from drive "
              f"{current_drive or '?'} in {KITTI_DIR} (need {num}).")
        print("  Use --force to re-download, or --drive <ID> to switch sequences.")
        return

    if drive not in KNOWN_DRIVES:
        print(f"ERROR: Drive '{drive}' is not a known KITTI raw drive.\n")
        _print_available_drives()
        print(f"\nUsage: python download_kitti.py --drive 0014")
        sys.exit(1)

    drive_name = f"{RAW_DATE}_drive_{drive}_sync"
    url = DRIVE_URL_TEMPLATE.format(date=RAW_DATE, drive=drive)
    expected_frames = KNOWN_DRIVES[drive]

    if num > expected_frames:
        print(f"WARNING: Drive {drive} has only {expected_frames} frames, "
              f"but --num {num} was requested. Will use all available frames.")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "drive.zip"

        print(f"Downloading KITTI raw drive {drive} ({drive_name}) ...")
        print(f"  URL: {url}")
        print(f"  Expected frames: {expected_frames}")
        try:
            urlretrieve(url, str(zip_path), reporthook=_reporthook)
        except HTTPError as e:
            print(f"\n\nERROR: Download failed — HTTP {e.code} {e.reason}")
            if e.code == 404:
                print(f"  Drive '{drive}' was not found on the KITTI server.")
                print(f"  This may indicate the drive has been removed.\n")
                _print_available_drives()
            sys.exit(1)
        print()

        with open(zip_path, "rb") as f:
            magic = f.read(4)
        if magic != b"PK\x03\x04":
            with open(zip_path, "rb") as f:
                preview = f.read(500).decode("utf-8", errors="replace")
            print(f"ERROR: Downloaded file is not a valid zip.\n"
                  f"Server response:\n{preview}")
            sys.exit(1)

        print("Extracting ...")
        extract_dir = Path(tmp) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(str(extract_dir))

        pattern = str(extract_dir / "**" / "image_02" / "data")
        candidates = _glob.glob(pattern, recursive=True)
        if not candidates:
            print("ERROR: Could not find image_02/data/ in the extracted zip.")
            for root, dirs, _ in os.walk(str(extract_dir)):
                depth = root.replace(str(extract_dir), "").count(os.sep)
                if depth < 4:
                    print(f"  {'  ' * depth}{os.path.basename(root)}/")
            sys.exit(1)

        img_dir = Path(candidates[0])
        all_frames = sorted(img_dir.glob("*.png"))
        if not all_frames:
            print(f"ERROR: No PNG frames found in {img_dir}")
            sys.exit(1)

        selected = all_frames[:num]
        print(f"Found {len(all_frames)} sequential frames, "
              f"selecting first {len(selected)}.")

        if KITTI_DIR.exists():
            shutil.rmtree(KITTI_DIR)
        KITTI_DIR.mkdir(parents=True, exist_ok=True)

        for src in selected:
            shutil.copy2(str(src), str(KITTI_DIR / src.name))

        _write_meta(drive, len(selected))

    print(f"\n{len(selected)} sequential KITTI frames saved to {KITTI_DIR}/")
    print(f"  Drive: {drive_name}")
    print(f"  Frames: {selected[0].name} ... {selected[-1].name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download sequential KITTI raw drive frames for BEV Fusion")
    parser.add_argument("--drive", type=str, default="0014",
                        help="KITTI drive ID (default: 0014). "
                             "Use --list to see all available drives.")
    parser.add_argument("--num", type=int, default=DEFAULT_NUM,
                        help=f"Number of sequential frames to keep (default {DEFAULT_NUM})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if frames already exist")
    parser.add_argument("--list", action="store_true",
                        help="List all available KITTI drives and exit")
    args = parser.parse_args()

    if args.list:
        _print_available_drives()
        return

    download_and_extract(drive=args.drive, num=args.num, force=args.force)


if __name__ == "__main__":
    main()
