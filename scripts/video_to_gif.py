#!/usr/bin/env python3
"""
Convert the first N seconds of a video to an animated GIF for README (GitHub shows GIFs inline).
Uses OpenCV + PIL. Keep duration and resolution low to avoid huge files.

Usage (from repo root):
  python scripts/video_to_gif.py -i assets/vis_demo_10mb.mp4 -o assets/vis_demo.gif --seconds 4 --fps 5 --width 480
"""

import argparse
import os
import sys

try:
    import cv2
except ImportError:
    print("opencv-python is required: pip install opencv-python")
    sys.exit(1)
try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert video to animated GIF (short clip for README)")
    parser.add_argument("--input", "-i", default="assets/vis_demo_10mb.mp4", help="Input video")
    parser.add_argument("--output", "-o", default="assets/vis_demo.gif", help="Output GIF")
    parser.add_argument("--seconds", "-s", type=float, default=4.0, help="Duration in seconds (default 4)")
    parser.add_argument("--fps", type=float, default=5.0, help="GIF frame rate (default 5; lower = smaller file)")
    parser.add_argument("--width", "-w", type=int, default=480, help="Output width (default 480; height auto)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames (overrides --seconds if set)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input not found:", args.input)
        return 1

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Could not open:", args.input)
        return 1

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_frames is not None:
        n_frames = min(args.max_frames, total_frames)
    else:
        n_frames = min(int(args.seconds * vid_fps), total_frames)

    # Sample to get args.fps in output: take every (vid_fps/args.fps) frame
    step = max(1, int(round(vid_fps / args.fps)))
    out_frames = []
    out_w = args.width
    out_h = int(round(h * out_w / w)) if w else out_w
    out_h = out_h - (out_h % 2)  # even height for some codecs

    print(f"Input: {args.input} — {total_frames} frames @ {vid_fps:.1f} fps, {w}x{h}")
    print(f"Taking every {step} frame, max {n_frames} frames -> ~{min(n_frames // step, n_frames)} GIF frames @ {args.fps} fps, {out_w}x{out_h}")

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % step != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (out_w, out_h) != (w, h):
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out_frames.append(Image.fromarray(rgb))
    cap.release()

    if not out_frames:
        print("No frames read")
        return 1

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save as GIF (PIL uses 256 colors by default for GIF)
    duration_ms = int(1000 / args.fps)
    out_frames[0].save(
        args.output,
        save_all=True,
        append_images=out_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Output: {args.output} ({size_mb:.2f} MB, {len(out_frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
