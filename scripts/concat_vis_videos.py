#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate all vis_v*.mp4 videos from a folder (e.g. vis_results) into a single
video file. Sorts by video number (vis_v1, vis_v2, ...) so the sequence is 1, 2, 3, ...
Resizes every frame to a fixed size to handle mixed dimensions and keep file size down.

Usage:
  python concat_vis_videos.py [--input vis_results] [--output vis_results_combined.mp4]
  python concat_vis_videos.py --input vis_results --output combined.mp4 --width 640 --height 360

Requires: opencv-python (pip install opencv-python)
"""

import argparse
import os
import re
import sys

import numpy as np
try:
    import cv2
except ImportError:
    print("opencv-python is required: pip install opencv-python")
    sys.exit(1)


def _natural_sort_key(path):
    """Sort key: extract integer from vis_vN.mp4 so we get 1, 2, 3, ..., 10, 11 not 1, 10, 100, 2."""
    basename = os.path.basename(path)
    m = re.search(r"vis_v(\d+)\.mp4$", basename, re.IGNORECASE)
    return (int(m.group(1)), path) if m else (0, path)


def collect_videos(input_dir, pattern=None):
    """Return list of .mp4 paths in input_dir, sorted by vis_v number."""
    if pattern is None:
        pattern = re.compile(r"^vis_v\d+\.mp4$", re.IGNORECASE)
    paths = []
    for name in os.listdir(input_dir):
        if pattern.match(name):
            paths.append(os.path.join(input_dir, name))
    paths.sort(key=_natural_sort_key)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Concatenate vis_v*.mp4 into one video")
    parser.add_argument("--input", "-i", default="vis_results", help="Folder containing vis_v*.mp4 files")
    parser.add_argument("--output", "-o", default="vis_results_combined.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=640, help="Output width (default 640)")
    parser.add_argument("--height", type=int, default=360, help="Output height (default 360)")
    parser.add_argument("--fps", type=float, default=10.0, help="Output FPS (default 10, lower = smaller file)")
    parser.add_argument("--pattern", default=None, help="Optional: only include files matching regex (e.g. vis_v\\d+\\.mp4)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print("Input directory not found:", args.input)
        return 1

    paths = collect_videos(args.input)
    if not paths:
        print("No vis_v*.mp4 files found in", args.input)
        return 1

    print(f"Found {len(paths)} videos, output {args.width}x{args.height} @ {args.fps} fps -> {args.output}")

    out_w = args.width
    out_h = args.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
    if not out_writer.isOpened():
        print("Could not create output video:", args.output)
        return 1

    total_frames = 0
    for i, path in enumerate(paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  Skip (cannot open): {path}")
            continue
        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to fixed size (uniform scale to fit, then pad or crop to exact size)
            h, w = frame.shape[:2]
            scale = min(out_w / w, out_h / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            if nw <= 0 or nh <= 0:
                nw, nh = out_w, out_h
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            # Center on canvas (letterbox)
            out_frame = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
            out_frame[:] = (30, 30, 30)  # dark gray background (BGR)
            y0 = (out_h - nh) // 2
            x0 = (out_w - nw) // 2
            out_frame[y0 : y0 + nh, x0 : x0 + nw] = resized
            out_writer.write(out_frame)
            n += 1
            total_frames += 1
        cap.release()
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(paths)}: {os.path.basename(path)} ({n} frames)")

    out_writer.release()
    print(f"Done: {total_frames} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
