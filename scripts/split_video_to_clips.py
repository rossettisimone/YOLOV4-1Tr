#!/usr/bin/env python3
"""
Split a video into N equal clips, each written at the given fps (e.g. 4 fps).
Uses OpenCV. Run from repo root.

Usage:
  python scripts/split_video_to_clips.py -i assets/vis_results_combined_under100mb.mp4 -o assets/clips_4fps -n 50 --fps 4
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
    parser = argparse.ArgumentParser(description="Split video into N clips at given fps")
    parser.add_argument("--input", "-i", default="assets/vis_results_combined_under100mb.mp4", help="Input video")
    parser.add_argument("--output-dir", "-o", default="assets/clips_4fps", help="Output directory for clip_001.mp4 ...")
    parser.add_argument("--num-clips", "-n", type=int, default=50, help="Number of clips (default 50)")
    parser.add_argument("--fps", type=float, default=4.0, help="Output frame rate (default 4)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input not found:", args.input)
        return 1

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open:", args.input)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_clips = max(1, args.num_clips)
    frames_per_clip = total // n_clips
    if frames_per_clip < 1:
        print("Not enough frames for", n_clips, "clips (total frames:", total, ")")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"Input: {args.input} — {total} frames, {w}x{h}")
    print(f"Splitting into {n_clips} clips of ~{frames_per_clip} frames each, output {args.fps} fps -> {args.output_dir}/")

    for c in range(n_clips):
        start = c * frames_per_clip
        end = (c + 1) * frames_per_clip if c < n_clips - 1 else total
        out_path = os.path.join(args.output_dir, f"clip_{c+1:03d}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
        if not out.isOpened():
            print("  Failed to create", out_path)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  {c+1:3d}/{n_clips}  clip_{c+1:03d}.mp4  ({end - start} frames, {size_kb:.0f} KB)")

    cap.release()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
