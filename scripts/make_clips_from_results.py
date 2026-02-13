#!/usr/bin/env python3
"""
Pipeline: overlay tracking results (e.g. results-test-heuristics.json) onto test-set frames,
concatenate the per-video outputs, then split into N clips at a given fps.

Requires the YouTube-VOS test set on disk (--frames-dir). Run from repo root.

Usage:
  python scripts/make_clips_from_results.py --frames-dir path/to/YouTubeVIS21/test
  python scripts/make_clips_from_results.py --frames-dir path/to/test --results results-test-heuristics.json -o vis_results_heuristics -n 50 --fps 4
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Overlay results -> concat -> split into N clips")
    parser.add_argument("--frames-dir", required=True, help="Test set root (folder containing JPEGImages/)")
    parser.add_argument("--results", default="results-test-heuristics.json", help="Tracking JSON (default: results-test-heuristics.json)")
    parser.add_argument("--annotations", default="pred_test_instances.json", help="Frame-order JSON (default: pred_test_instances.json)")
    parser.add_argument("--output-base", "-o", default="vis_results_heuristics", help="Output folder for overlay videos and combined name base")
    parser.add_argument("--num-clips", "-n", type=int, default=50, help="Number of clips to split into (default 50)")
    parser.add_argument("--fps", type=float, default=4.0, help="Output clip fps (default 4)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    if not os.path.isfile(args.results):
        print("Results file not found:", args.results)
        return 1
    if not os.path.isfile(args.annotations):
        print("Annotations file not found:", args.annotations)
        return 1
    jpeg = os.path.join(args.frames_dir, "JPEGImages")
    if not os.path.isdir(jpeg):
        print("--frames-dir must contain JPEGImages/. Not found:", jpeg)
        return 1

    overlay_dir = args.output_base
    combined_mp4 = f"{args.output_base}_combined.mp4"
    clips_dir = f"assets/clips_4fps_{os.path.basename(args.output_base)}"

    # 1) Overlay: results JSON -> per-video overlay videos
    print("Step 1: Overlaying results onto test-set frames...")
    r = subprocess.run([
        sys.executable, "scripts/overlay_vis_on_video.py",
        "--from-dataset", "--frames-dir", args.frames_dir,
        "--results", args.results,
        "--annotations", args.annotations,
        "--all-videos", "--output", overlay_dir,
    ], cwd=repo_root)
    if r.returncode != 0:
        print("Overlay failed.")
        return r.returncode

    # 2) Concat: vis_v*.mp4 -> one combined video
    print("Step 2: Concatenating overlay videos...")
    r = subprocess.run([
        sys.executable, "scripts/concat_vis_videos.py",
        "--input", overlay_dir, "--output", combined_mp4,
    ], cwd=repo_root)
    if r.returncode != 0:
        print("Concat failed.")
        return r.returncode

    if not os.path.isfile(combined_mp4):
        print("Combined video not found:", combined_mp4)
        return 1

    # 3) Split: combined -> N clips at fps
    print("Step 3: Splitting into", args.num_clips, "clips at", args.fps, "fps...")
    r = subprocess.run([
        sys.executable, "scripts/split_video_to_clips.py",
        "-i", combined_mp4, "-o", clips_dir,
        "-n", str(args.num_clips), "--fps", str(args.fps),
    ], cwd=repo_root)
    if r.returncode != 0:
        print("Split failed.")
        return r.returncode

    print("Done. Clips saved in", clips_dir + "/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
