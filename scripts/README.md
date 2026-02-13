# Scripts

Standalone utilities for VIS visualization and evaluation. Run from the **repository root**.  
Generated outputs (`vis_results*/`, `*_combined.mp4`, `assets/clips_4fps*/`) are in `.gitignore`; regenerate as needed.

| Script | Purpose |
|--------|--------|
| `overlay_vis_on_video.py` | Overlay tracking masks from JSON onto video (dataset or single file) |
| `concat_vis_videos.py` | Concatenate `vis_v*.mp4` from a folder into one video |
| `shrink_vis_video.py` | Trim video by fraction or `--max-mb` (e.g. 100 or 10 for GitHub) |
| `video_to_gif.py` | Convert first N seconds of video to GIF for README (plays inline on GitHub) |
| `split_video_to_clips.py` | Split a video into N equal clips at a given fps (e.g. 50 clips at 4 fps) |
| `make_clips_from_results.py` | Pipeline: overlay results JSON → concat → N clips. Requires test set (`--frames-dir`). |
| `evaluate.py` | YouTube-VOS evaluation with pycocotools |

**Demo / README pipeline** (GIF and short video for README):

```bash
python scripts/overlay_vis_on_video.py --from-dataset --frames-dir /path/to/test --all-videos --output vis_results
python scripts/concat_vis_videos.py --input vis_results --output vis_results_combined.mp4
python scripts/shrink_vis_video.py --max-mb 100
python scripts/shrink_vis_video.py -i assets/vis_results_combined_under100mb.mp4 --max-mb 10 -o assets/vis_demo_10mb.mp4
python scripts/video_to_gif.py -i assets/vis_demo_10mb.mp4 -o assets/vis_demo.gif --seconds 26 --fps 4 --width 480
```

**50 clips at 4 fps** (from existing combined video or from a results JSON):

```bash
python scripts/split_video_to_clips.py -i assets/vis_results_combined_under100mb.mp4 -o assets/clips_4fps -n 50 --fps 4
# Or from results-test-heuristics.json (needs test set):
python scripts/make_clips_from_results.py --frames-dir /path/to/YouTubeVIS21/test --results results-test-heuristics.json -o vis_results_heuristics -n 50 --fps 4
```
