# Scripts

Standalone utilities for VIS visualization and evaluation. Run from the **repository root**.

| Script | Purpose |
|--------|--------|
| `overlay_vis_on_video.py` | Overlay tracking masks from JSON onto video (dataset or single file) |
| `concat_vis_videos.py` | Concatenate `vis_v*.mp4` from a folder into one video |
| `shrink_vis_video.py` | Trim combined video to first half (under 100 MB for GitHub) |
| `evaluate.py` | YouTube-VOS evaluation with pycocotools |

Example (from repo root):

```bash
python scripts/overlay_vis_on_video.py --from-dataset --frames-dir /path/to/test --all-videos --output vis_results
python scripts/concat_vis_videos.py --input vis_results --output vis_results_combined.mp4
python scripts/shrink_vis_video.py   # writes assets/vis_results_combined_under100mb.mp4
```
