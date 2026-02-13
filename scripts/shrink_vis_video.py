#!/usr/bin/env python3
"""
Cut or trim vis_results_combined.mp4 to stay under a size limit (e.g. GitHub 100 MB or 10 MB).
Uses OpenCV (no ffmpeg required).

Usage:
  python shrink_vis_video.py --max-mb 100   # default: first half, output assets/...under100mb.mp4
  python shrink_vis_video.py --max-mb 10 -o assets/vis_demo_10mb.mp4
  python shrink_vis_video.py --fraction 0.5 -o out.mp4
"""

import argparse
import os
import sys

try:
    import cv2
except ImportError:
    print("opencv-python is required: pip install opencv-python")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Trim video to stay under a size limit (e.g. 100 MB or 10 MB)")
    parser.add_argument("--input", "-i", default="vis_results_combined.mp4", help="Input video")
    parser.add_argument("--output", "-o", default=None, help="Output path (default depends on --max-mb)")
    parser.add_argument("--fraction", "-f", type=float, default=None, help="Fraction of video to keep (e.g. 0.5 = first half)")
    parser.add_argument("--max-mb", "-m", type=float, default=None, help="Target max size in MB (trim to fit; uses input file size to estimate frames)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input not found:", args.input)
        return 1

    input_size_mb = os.path.getsize(args.input) / (1024 * 1024)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open:", args.input)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_mb is not None:
        # Estimate frames to keep so output stays under args.max_mb (assume similar bytes/frame)
        target_mb = args.max_mb * 0.75  # stay safely under (encoding overhead varies)
        keep = max(1, min(total, int(total * target_mb / input_size_mb)))
        if args.output is None:
            args.output = f"assets/vis_results_combined_under{int(args.max_mb)}mb.mp4"
    elif args.fraction is not None:
        keep = max(1, int(total * args.fraction))
        if args.output is None:
            args.output = "assets/vis_results_combined_under100mb.mp4"
    else:
        args.fraction = 0.5
        keep = max(1, int(total * 0.5))
        args.output = args.output or "assets/vis_results_combined_under100mb.mp4"

    pct = 100 * keep / total if total else 0
    print(f"Input: {args.input} — {total} frames, {fps:.1f} fps, {w}x{h} ({input_size_mb:.1f} MB)")
    print(f"Writing first {keep} frames ({pct:.1f}%) -> {args.output}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not out.isOpened():
        print("Could not create output:", args.output)
        return 1

    n = 0
    while n < keep:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{keep}")

    cap.release()
    out.release()
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Output: {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
