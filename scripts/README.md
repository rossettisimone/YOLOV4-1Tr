# Scripts

Standalone utilities for VIS visualization and evaluation. Run from the **repository root**.

| Script | Purpose |
|--------|--------|
| `overlay_vis_on_video.py` | Overlay tracking masks from JSON onto video (dataset or single file) |
| `concat_vis_videos.py` | Concatenate `vis_v*.mp4` from a folder into one video |
| `shrink_vis_video.py` | Trim video by fraction or `--max-mb` (e.g. 100 or 10 for GitHub) |
| `evaluate.py` | YouTube-VOS evaluation with pycocotools |

Example (from repo root):

```bash
python scripts/overlay_vis_on_video.py --from-dataset --frames-dir /path/to/test --all-videos --output vis_results
python scripts/concat_vis_videos.py --input vis_results --output vis_results_combined.mp4
python scripts/shrink_vis_video.py --max-mb 100   # -> assets/vis_results_combined_under100mb.mp4
python scripts/shrink_vis_video.py -i assets/vis_results_combined_under100mb.mp4 --max-mb 10 -o assets/vis_demo_10mb.mp4
```
