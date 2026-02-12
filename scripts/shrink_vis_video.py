#!/usr/bin/env python3
"""
Cut the first half of vis_results_combined.mp4 to get under GitHub's 100 MB limit.
Uses OpenCV (no ffmpeg required). Output: vis_results_combined_under100mb.mp4

Usage: python shrink_vis_video.py [--input vis_results_combined.mp4] [--output vis_results_combined_under100mb.mp4]
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
    parser = argparse.ArgumentParser(description="Cut first half of video for GitHub 100MB limit")
    parser.add_argument("--input", "-i", default="vis_results_combined.mp4", help="Input video")
    parser.add_argument("--output", "-o", default="assets/vis_results_combined_under100mb.mp4", help="Output video (first half)")
    parser.add_argument("--fraction", "-f", type=float, default=0.5, help="Fraction to keep (default 0.5 = first half)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input not found:", args.input)
        return 1

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open:", args.input)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    keep = max(1, int(total * args.fraction))
    print(f"Input: {args.input} — {total} frames, {fps:.1f} fps, {w}x{h}")
    print(f"Writing first {keep} frames ({100*args.fraction:.0f}%) -> {args.output}")

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
