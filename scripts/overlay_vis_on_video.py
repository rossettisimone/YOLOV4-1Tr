#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay instance segmentation and tracking results from JSON onto a video.
Uses results_test_new_graph.json (tracking results).
Output: video with colored instance masks and labels overlaid.

Modes:
  1) Single video / all-videos from one input video: uses --video (e.g. test_set.mp4).
     Frame order is consecutive; use only if your input video was built in the same
     order as video_id in the JSON (otherwise masks will be misaligned).

  2) From dataset (correct alignment): use --from-dataset --frames-dir <path>.
     Loads frames from the YouTube-VOS test set using pred_test_instances.json so
     frame order and identity match the predictions. Requires the test set on disk
     (download from https://youtube-vos.org/dataset/). frames-dir = folder that
     contains JPEGImages/ (e.g. path/to/YouTubeVIS21/test).

Usage:
  python overlay_vis_on_video.py [--video test_set.mp4] [--results ...] [--output ...]
  python overlay_vis_on_video.py --from-dataset --frames-dir path/to/test --results ... --annotations pred_test_instances.json
  python overlay_vis_on_video.py --from-dataset --all-videos --frames-dir path/to/test

Requires: opencv-python (pip install opencv-python). No TensorFlow dependency.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    print("opencv-python is required: pip install opencv-python")
    sys.exit(1)

# Standalone RLE decoding (same as utils.rle_decoding) to avoid importing utils (TensorFlow)
# size in JSON is typically [height, width]; set size_as_wh=True if your JSON uses [width, height]
def rle_decoding(rle, size_as_wh=False):
    raw = rle.get("size", [720, 1280])
    if size_as_wh:
        w, h = raw[0], raw[1]  # [width, height]
    else:
        h, w = raw[0], raw[1]  # [height, width]
    rle_arr = rle.get("counts", [0])
    rle_arr = np.cumsum(rle_arr)
    indices = []
    extend = indices.extend
    list(map(extend, map(lambda s, e: range(s, e), rle_arr[0::2], rle_arr[1::2])))
    binary_mask = np.zeros(h * w, dtype=np.uint8)
    binary_mask[indices] = 1
    return binary_mask.reshape((w, h)).T  # (h, w)

# YouTube-VIS 2021 class names (id -> name)
CLASS_YTVIS21 = {
    1: "airplane", 2: "bear", 3: "bird", 4: "boat", 5: "car", 6: "cat", 7: "cow", 8: "deer",
    9: "dog", 10: "duck", 11: "earless_seal", 12: "elephant", 13: "fish", 14: "flying_disc",
    15: "fox", 16: "frog", 17: "giant_panda", 18: "giraffe", 19: "horse", 20: "leopard",
    21: "lizard", 22: "monkey", 23: "motorbike", 24: "mouse", 25: "parrot", 26: "person",
    27: "rabbit", 28: "shark", 29: "skateboard", 30: "snake", 31: "snowboard", 32: "squirrel",
    33: "surfboard", 34: "tennis_racket", 35: "tiger", 36: "train", 37: "truck", 38: "turtle",
    39: "whale", 40: "zebra",
}


# Distinct colors for instances (BGR for OpenCV)
def _get_instance_colors(n):
    np.random.seed(42)
    colors = []
    for i in range(max(n, 1)):
        hue = (i * 137.508) % 360  # golden angle for spread
        # Convert HSV to RGB then to BGR
        h = hue / 360.0
        s, v = 0.8, 0.95
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors.append((b, g, r))  # BGR
    return colors


def load_tracking_results(path):
    with open(path) as f:
        tracks = json.load(f)
    by_video = defaultdict(list)
    for t in tracks:
        by_video[t["video_id"]].append(t)
    return dict(by_video)


def load_annotations_frame_order(annotations_path):
    """
    Load pred_test_instances.json (or similar) and return
    video_id -> [file_name, ...] sorted by num_sequence so frame index f
    matches the f-th frame in the tracking segmentations.
    """
    with open(annotations_path) as f:
        data = json.load(f)
    ann = data.get("annotations", data) if isinstance(data, dict) else data
    by_vid = defaultdict(list)
    for a in ann:
        vid = a["video_id"]
        seq = a.get("num_sequence", -1)
        fname = a.get("file_name", "")
        by_vid[vid].append((seq, fname))
    out = {}
    for vid, pairs in by_vid.items():
        pairs.sort(key=lambda x: x[0])
        out[vid] = [fname for _, fname in pairs]
    return out


def find_video_by_frame_count(by_video, frame_count):
    for vid, track_list in sorted(by_video.items()):
        if not track_list:
            continue
        n_frames = len(track_list[0]["segmentations"])
        if n_frames == frame_count:
            return vid, track_list
    return None, None


def rle_to_mask(rle, target_h, target_w, preserve_aspect=True, size_as_wh=False):
    """
    Decode RLE to binary mask and resize to (target_h, target_w).
    If preserve_aspect=True, scale mask uniformly to fit inside the frame and center it,
    avoiding vertical/horizontal stretch misalignment.
    """
    if rle is None:
        return None
    mask = rle_decoding(rle, size_as_wh=size_as_wh)  # (mask_h, mask_w)
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    if mask_h == target_h and mask_w == target_w:
        return mask
    if preserve_aspect:
        # Same scaling in x and y so masks aren't "zoomed" or vertically misaligned
        scale = min(target_w / mask_w, target_h / mask_h)
        new_w = max(1, round(mask_w * scale))
        new_h = max(1, round(mask_h * scale))
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (new_w, new_h), Image.NEAREST
            )
        )
        # Center on (target_h, target_w) canvas
        out = np.zeros((target_h, target_w), dtype=mask_resized.dtype)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        out[y0 : y0 + new_h, x0 : x0 + new_w] = mask_resized
        return out
    else:
        return np.array(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (target_w, target_h), Image.NEAREST
            )
        )


def overlay_frame(frame_bgr, tracks_for_video, frame_idx, class_dict, colors, alpha=0.5, preserve_aspect=True, size_as_wh=False):
    """
    Overlay instance masks and labels on one frame.
    frame_bgr: (H, W, 3) BGR uint8
    tracks_for_video: list of track dicts (video_id, category_id, segmentations, score)
    frame_idx: index into segmentations list
    """
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy().astype(np.float32)
    centroids = []  # (cx, cy, label) for text later

    for i, track in enumerate(tracks_for_video):
        segs = track["segmentations"]
        if frame_idx >= len(segs):
            continue
        rle = segs[frame_idx]
        mask = rle_to_mask(rle, h, w, preserve_aspect=preserve_aspect, size_as_wh=size_as_wh)
        if mask is None or not np.any(mask > 0):
            continue
        color = np.array(colors[i % len(colors)], dtype=np.float32)
        where_mask = (mask > 0)[:, :, np.newaxis]
        out = np.where(where_mask, alpha * color + (1 - alpha) * out, out)

        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            cy, cx = int(ys.mean()), int(xs.mean())
            cat_id = track.get("category_id", 0)
            cat_name = class_dict.get(cat_id, str(cat_id))[:12]
            centroids.append((cx, cy, f"#{i+1} {cat_name}"))

    out = np.clip(out, 0, 255).astype(np.uint8)
    for (cx, cy, label) in centroids:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_overlay_video(cap, track_list, video_id, out_path, fps, w, h, class_dict, colors, args, preserve_aspect):
    """Write one overlay video for the given video_id (segment of cap). Returns frames written."""
    n_frames = len(track_list[0]["segmentations"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out_writer.isOpened():
        return 0
    written = 0
    for f in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_overlay = overlay_frame(
            frame, track_list, f, class_dict, colors, alpha=args.alpha,
            preserve_aspect=preserve_aspect, size_as_wh=args.rle_size_wh
        )
        cv2.putText(frame_overlay, f"YOLOV4+1Tr VIS | video_id={video_id} | frame {f+1}/{n_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_overlay, f"YOLOV4+1Tr VIS | video_id={video_id} | frame {f+1}/{n_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        out_writer.write(frame_overlay)
        written += 1
    out_writer.release()
    return written


def _write_overlay_video_from_frames(frame_paths, track_list, video_id, out_path, fps, class_dict, colors, args, preserve_aspect):
    """
    Write one overlay video by loading frames from disk (correct order from annotations).
    frame_paths: list of paths to images (same order as segmentations).
    Returns (frames_written, error_message or None).
    """
    n_frames = len(track_list[0]["segmentations"])
    if len(frame_paths) != n_frames:
        return 0, f"frame count mismatch: {len(frame_paths)} files vs {n_frames} segmentations"
    if not frame_paths:
        return 0, "no frame paths"
    # Load first frame to get size
    first = np.array(Image.open(frame_paths[0]).convert("RGB"))
    if len(first.shape) == 2:
        first = np.stack([first] * 3, axis=-1)
    h, w = first.shape[0], first.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out_writer.isOpened():
        return 0, f"could not create {out_path}"
    written = 0
    for f, path in enumerate(frame_paths):
        if not os.path.isfile(path):
            return written, f"missing frame: {path}"
        img = np.array(Image.open(path).convert("RGB"))
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame_overlay = overlay_frame(
            frame_bgr, track_list, f, class_dict, colors, alpha=args.alpha,
            preserve_aspect=preserve_aspect, size_as_wh=args.rle_size_wh
        )
        cv2.putText(frame_overlay, f"YOLOV4+1Tr VIS | video_id={video_id} | frame {f+1}/{n_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_overlay, f"YOLOV4+1Tr VIS | video_id={video_id} | frame {f+1}/{n_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        out_writer.write(frame_overlay)
        written += 1
    out_writer.release()
    return written, None


def main():
    parser = argparse.ArgumentParser(description="Overlay VIS tracking results on video")
    parser.add_argument("--video", default="test_set.mp4", help="Input video path")
    parser.add_argument("--results", default="results_test_new_graph.json", help="Tracking JSON (results_test_new_graph.json)")
    parser.add_argument("--output", default="test_set_vis_results.mp4", help="Output video path (or output dir when --all-videos)")
    parser.add_argument("--video-id", type=int, default=None, help="Force this video_id from JSON (optional)")
    parser.add_argument("--all-videos", action="store_true", help="Process all video_ids: one output per video, using consecutive segments of input video")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity 0-1")
    parser.add_argument("--no-preserve-aspect", action="store_true", help="Stretch mask to frame (may cause vertical/horizontal misalignment)")
    parser.add_argument("--rle-size-wh", action="store_true", help="Interpret RLE size as [width, height] instead of [height, width]")
    # From-dataset: load frames from disk (correct order from annotations)
    parser.add_argument("--from-dataset", action="store_true", help="Load frames from YouTube-VOS test set; use with --frames-dir and --annotations for correct alignment")
    parser.add_argument("--frames-dir", default="", help="Root dir containing JPEGImages/ (e.g. path/to/YouTubeVIS21/test). Used with --from-dataset.")
    parser.add_argument("--annotations", default="pred_test_instances.json", help="Per-frame annotations JSON (for --from-dataset frame order)")
    parser.add_argument("--fps", type=float, default=6.0, help="Output video FPS when using --from-dataset (default 6)")
    args = parser.parse_args()
    preserve_aspect = not args.no_preserve_aspect

    # Load tracking results
    by_video = load_tracking_results(args.results)
    if not by_video:
        print("No tracks in", args.results)
        return 1

    if args.from_dataset:
        if not args.frames_dir or not os.path.isdir(args.frames_dir):
            print("--from-dataset requires --frames-dir pointing to the test root (folder containing JPEGImages/).")
            print("Download the YouTube-VOS 2021 test set from https://youtube-vos.org/dataset/ and set --frames-dir accordingly.")
            return 1
        if not os.path.isfile(args.annotations):
            print("Annotations file not found:", args.annotations)
            return 1
        vid_to_files = load_annotations_frame_order(args.annotations)
        out_dir = args.output.strip().rstrip("/\\")
        if out_dir.endswith(".mp4") or not out_dir:
            out_dir = "vis_results"
        out_dir = out_dir or "vis_results"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        video_ids = sorted(by_video.keys()) if args.all_videos else ([args.video_id] if args.video_id is not None else [min(by_video.keys())])
        fps = args.fps
        written_count = 0
        for video_id in video_ids:
            if video_id not in by_video:
                continue
            if video_id not in vid_to_files:
                print(f"  Skip video_id={video_id} (not in annotations)")
                continue
            frame_paths = [os.path.normpath(os.path.join(args.frames_dir, fn)) for fn in vid_to_files[video_id]]
            track_list = by_video[video_id]
            if not frame_paths or not os.path.isfile(frame_paths[0]):
                print(f"  Skip video_id={video_id} (frames not found; is the test set at --frames-dir?) Example path: {frame_paths[0] if frame_paths else '?'}")
                continue
            out_path = os.path.join(out_dir, f"vis_v{video_id}.mp4")
            colors = _get_instance_colors(len(track_list))
            class_dict = CLASS_YTVIS21
            n_written, err = _write_overlay_video_from_frames(
                frame_paths, track_list, video_id, out_path, fps, class_dict, colors, args, preserve_aspect
            )
            if err:
                print(f"  video_id={video_id} error: {err}")
                continue
            written_count += 1
            print(f"  video_id={video_id} -> {out_path} ({n_written} frames)")
        print(f"Saved {written_count} videos in {out_dir}/")
        return 0

    # Open input video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Could not open video:", args.video)
        return 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video} -> {total_frames} frames, {w}x{h}, {fps} fps")

    if args.all_videos:
        # One output per video_id, using consecutive chunks of input video
        out_dir = args.output.strip()
        if out_dir.endswith(".mp4"):
            out_dir = "vis_results"
        out_dir = out_dir.rstrip("/\\") or "."
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        video_ids = sorted(by_video.keys())
        print(f"Processing {len(video_ids)} videos, writing to {out_dir}/")
        frame_offset = 0
        written = 0
        for video_id in video_ids:
            track_list = by_video[video_id]
            n_frames = len(track_list[0]["segmentations"])
            if frame_offset + n_frames > total_frames:
                print(f"  Skip video_id={video_id} (need {n_frames} frames, only {total_frames - frame_offset} left)")
                break
            out_path = os.path.join(out_dir, f"vis_v{video_id}.mp4")
            colors = _get_instance_colors(len(track_list))
            class_dict = CLASS_YTVIS21
            n_written = _write_overlay_video(cap, track_list, video_id, out_path, fps, w, h, class_dict, colors, args, preserve_aspect)
            frame_offset += n_frames
            written += 1
            print(f"  video_id={video_id} -> {out_path} ({n_written} frames)")
        cap.release()
        print(f"Saved {written} videos in {out_dir}/")
        return 0

    # Single-video mode
    frame_count = total_frames
    if args.video_id is not None:
        if args.video_id not in by_video:
            print("video_id", args.video_id, "not found in JSON. Available (first 20):", list(by_video.keys())[:20])
            return 1
        video_id = args.video_id
        track_list = by_video[video_id]
        n_frames_json = len(track_list[0]["segmentations"])
        if n_frames_json != frame_count:
            print(f"Warning: JSON video_id {video_id} has {n_frames_json} frames, input video has {frame_count}. Using first {min(n_frames_json, frame_count)} frames.")
    else:
        video_id, track_list = find_video_by_frame_count(by_video, frame_count)
        if video_id is None:
            video_id = min(by_video.keys())
            track_list = by_video[video_id]
            n_frames_json = len(track_list[0]["segmentations"])
            print(f"No video in JSON with {frame_count} frames. Using video_id={video_id} ({n_frames_json} frames) and first {n_frames_json} frames of input.")
            frame_count = min(frame_count, n_frames_json)
        else:
            print(f"Matched video_id={video_id} ({len(track_list)} tracks, {frame_count} frames)")

    n_frames_to_write = min(frame_count, len(track_list[0]["segmentations"]))
    colors = _get_instance_colors(len(track_list))
    class_dict = CLASS_YTVIS21

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not out_writer.isOpened():
        print("Could not create output video:", args.output)
        cap.release()
        return 1

    for f in range(n_frames_to_write):
        ret, frame = cap.read()
        if not ret:
            break
        frame_overlay = overlay_frame(
            frame, track_list, f, class_dict, colors, alpha=args.alpha,
            preserve_aspect=preserve_aspect, size_as_wh=args.rle_size_wh
        )
        cv2.putText(
            frame_overlay, f"YOLOV4+1Tr VIS | frame {f+1}/{n_frames_to_write}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            frame_overlay, f"YOLOV4+1Tr VIS | frame {f+1}/{n_frames_to_write}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        out_writer.write(frame_overlay)
        if (f + 1) % 50 == 0 or f == 0:
            print(f"  Wrote frame {f+1}/{n_frames_to_write}")

    cap.release()
    out_writer.release()
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
