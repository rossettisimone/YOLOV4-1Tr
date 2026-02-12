<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&amp;height=220&amp;text=YOLOV4%2B1Tr&amp;fontSize=42&amp;fontAlign=50&amp;fontAlignY=40&amp;color=0:1e3a5f,100:c2410c&amp;fontColor=ffffff&amp;desc=Video%20Instance%20Segmentation%20%C2%B7%20YouTube-VOS%202021&amp;descSize=20&amp;descAlign=50&amp;descAlignY=62" alt="YOLOV4+1Tr banner" />
</p>

<p align="center">
  <img alt="Challenge YouTube-VOS 2021" src="https://img.shields.io/badge/Challenge-YouTube--VOS%202021%20%7C%20Track%202%20VIS-0ea5e9?style=for-the-badge" />
  <img alt="Lab Alcor Sapienza" src="https://img.shields.io/badge/Lab-Alcor%20%40%20Sapienza-7c3aed?style=for-the-badge" />
  <img alt="Model YOLOV4+1seg" src="https://img.shields.io/badge/Model-YOLOV4%2B1seg%20%C2%B7%20CSPDarknet53%20%C2%B7%20FPN-059669?style=for-the-badge" />
  <img alt="Framework TensorFlow" src="https://img.shields.io/badge/Framework-TensorFlow%202.3-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

**YOLOV4+1Tr** is a **video instance segmentation** pipeline built for the [3rd YouTube-VOS Challenge (CVPR 2021)](https://youtube-vos.org/challenge/2021/leaderboard/) — Track 2: Video Instance Segmentation. It extends **YOLOV4** with an extra FPN scale, a dedicated mask head, and a lightweight **greedy tracker** so you can detect, segment, and track object instances across video **without first-frame masks or prior bounding boxes**. One-stage detection + association: no Siamese re-detection, no template propagation — just a single forward pass per frame and class+mask IoU–based association.

**Team**: Simone Rossetti, Temirlan Zharbynbek, Fiora Pirri — [Alcor LAB](https://sites.google.com/diag.uniroma1.it/alcorlab-diag), Sapienza Università di Roma.  
**Technical report**: [VIS_10_Rossetti.pdf](VIS_10_Rossetti.pdf) (challenge submission).

---

## Demo — VIS in action

<p align="center">
  <strong>Detection · Segmentation · Tracking</strong><br>
  <em>Single-stage pipeline, no first-frame masks</em>
</p>

<p align="center">
  <video src="assets/vis_results_combined_under100mb.mp4" controls loop autoplay muted playsinline width="85%"></video>
</p>

<p align="center">
  <sub>Combined VIS results · YOLOV4+1Tr on YouTube-VOS 2021</sub>
</p>

---

## Quick navigation

- [Demo](#demo--vis-in-action)
- [What's in the box](#whats-in-the-box)
- [Architecture at a glance](#architecture-at-a-glance)
- [Method in short](#method-in-short)
- [Data & setup](#data--setup)
- [Quick start](#quick-start)
- [Results & challenge](#results--challenge)
- [Project layout](#project-layout)
- [References](#references)
- [Authors & license](#authors--license)

---

## What's in the box

- **YOLOV4+1** backbone & neck: **CSPDarknet53** + SPP + **extended FPN** (one extra level `b2/p2/n2`) for better small-object localization, with lateral connections and concatenation (YOLOV4-style blocks, not PANet-style addition).
- **Encode–decode proposal graph**: detection head with anchor-based regression and confidence; separate **feature embeddings** for mask prediction.
- **Instance segmentation head**: **Adaptive Feature Pooling** (AFP) over the four pyramid levels + small conv subnet for per-instance masks (Mask R-CNN–style ROI head, without a separate class/box refinement subnet).
- **Greedy tracker**: frame-to-frame association via **class+mask IoU** cost matrix, with **Gating**, **Best Friend**, and **Lonely Best Friend** heuristics to limit bad associations.
- **Multi-task training**: box regression (Smooth L1), confidence & mask (BCE), classification (cross-entropy), with **Automatic Loss Balancing** (learnable weights per layer and loss).
- **TensorFlow 2.3** implementation, distributed training with `MirroredStrategy`, and data loaders for **YouTube-VIS**-style annotations.

---

## Architecture at a glance

```
Input (416×416) → CSPDarknet53 (Mish) + SPP
       → Top-down FPN (b5→b4→b3→b2) + Bottom-up FPN (p2→p3→p4→p5)
       → Decode graph → Proposals (box, conf, class) + Feature embeddings
       → NMS → ROIAlign (AFP) on 4 levels → Mask subnet → Instance masks
       → Tracker: cost matrix (class+mask IoU) → Gating / Best Friend / Lonely Best Friend
```

Detection and segmentation are **single-stage** per frame; tracking is done by associating detections between consecutive frames. No first-frame mask or box is required; new instances can appear at any time.

---

## Method in short

- **Segmentation**: YOLOV4-style detection (CSPDarknet53, extended FPN, anchor-based proposals) plus an encode–decode graph that produces both **proposals** and **embedding maps**. Proposals go through NMS; then **Pyramid ROI Align** (AFP) crops and resizes from the four FPN levels; a small conv net predicts the binary mask per instance.
- **Tracking**: For frames \(t\) and \(t+1\), build a cost matrix \(A_{ij}\) = class+mask IoU between detection \(i\) at \(t\) and \(j\) at \(t+1\). Apply: (1) **Gating** (drop links below a threshold), (2) **Best Friend** (keep only mutual best matches), (3) **Lonely Best Friend** (keep only if gap between best and second-best is above a threshold). No Kalman, no re-ID embeddings — fast and simple.
- **Training**: Joint loss over pyramid levels and tasks (box, conf, class, mask) with learnable ALB weights; backbone can be frozen and then fine-tuned.

---

## Data & setup

- **Dataset**: [YouTube-VIS](https://youtube-vos.org/dataset/) (e.g. 2021 challenge splits). The code expects annotations and frames in a structure compatible with `loader_ytvos.py` (video → frames, instance IDs, categories, segmentations).
- **Environment**: Python 3.x, TensorFlow 2.3, TensorFlow Addons (e.g. `SGDW`). See `train.py` and `loader_ytvos.py` for dependencies.
- **Weights**: Pre-trained **CSPDarknet53** (YOLOV4) backbone is loaded from config; a challenge-era checkpoint is linked in the repo (e.g. `model.54--7.149.h5` — see below).

---

## Quick start

1. **Clone and install**
   ```bash
   git clone https://github.com/<your-username>/YOLOV4-1Tr.git
   cd YOLOV4-1Tr
   pip install tensorflow==2.3 tensorflow_addons  # and other deps (PIL, numpy, etc.)
   ```

2. **Data**
   - Download YouTube-VIS and set paths in config / `loader_ytvos.py`.
   - Ensure anchor priors match your data (e.g. `info/anchors_yt_16.txt` or run `clusters.py` for K-means anchors).

3. **Train**
   ```bash
   python train.py   # uses config (batch, LR, steps, etc.) and optional env.py
   ```
   Training uses `DataLoader` from `loader_ytvos`, `get_model()` from `model.py`, and callbacks for TensorBoard, checkpointing, and validation mAP/confusion matrix.

4. **Inference & tracking**
   - Run the model in `infer` mode to get boxes, confidence, class, and masks per frame.
   - Use the pipeline in `new_new_graph_tracking/` (e.g. `mainVIS_Track4.py`) to associate instances across frames and produce VIS-style submission (e.g. segmentations + instance IDs).

5. **Pre-trained weights**
   - Challenge submission weights (e.g. `model.54--7.149.h5`) are available at:  
     [Google Drive](https://drive.google.com/drive/folders/1oN-z71nxx1F4E7kQQKRxDdgNDz60K4g2?usp=sharing).

6. **Overlay tracking results on video** (`scripts/overlay_vis_on_video.py`)
   - **Correct alignment (recommended)**: use the YouTube-VOS test set so frame order matches the predictions. Download the [test set](https://youtube-vos.org/dataset/) and run:
     ```bash
     python scripts/overlay_vis_on_video.py --from-dataset --frames-dir /path/to/YouTubeVIS21/test \
       --results results_test_new_graph.json --annotations pred_test_instances.json \
       --all-videos --output vis_results
     ```
     `--frames-dir` must be the folder that contains `JPEGImages/` (e.g. the test split root). Frames are loaded in the order given by `num_sequence` in the annotations, so masks and video stay in sync.
   - **Single video**: `--from-dataset --frames-dir <test_root> --video-id 1 --output vis_results`
   - **Without the dataset**: you can overlay on a single input video (e.g. `test_set.mp4`) with `--video test_set.mp4`; only the first matching segment will be correct unless your input video was built in the same order as the JSON.
   - **Combine & shrink**: `scripts/concat_vis_videos.py` concatenates `vis_v*.mp4` into one file; `scripts/shrink_vis_video.py` trims it under 100 MB for GitHub.

---

## Results & challenge

This repository corresponds to the **SimoneRos** submission to the [YouTube-VOS 2021 Challenge — Track 2: Video Instance Segmentation](https://youtube-vos.org/challenge/2021/leaderboard/). It is a **research baseline** that prioritizes:

- **No first-frame assumption**: detection + segmentation + tracking in a unified pipeline without initial masks or boxes.
- **Real-time–oriented design**: one-stage detector (YOLOV4) + light greedy association.
- **Reproducibility**: full training and tracking code, clear architecture (extended FPN, AFP, ALB), and open implementation.

Leaderboard metrics (2021 VIS track) were modest compared to top entries; the main value here is the **architecture and pipeline** as a starting point for VIS and as a reference implementation of YOLOV4 extended to video instance segmentation with a simple tracker. For state-of-the-art numbers, see the top teams’ reports on the [official leaderboard](https://youtube-vos.org/challenge/2021/leaderboard/).

---

## Project layout

```
YOLOV4-1Tr/
├── model.py                  # YOLOV4+1 model: backbone, neck, decode, proposals, mask head
├── backbone.py               # CSPDarknet53 graph + weight loading
├── layers.py                 # FPN, decode, proposal, ROIAlign, mask_graph_AFP
├── train.py                  # Training loop, MirroredStrategy, callbacks
├── loader_ytvos.py           # YouTube-VIS data loader
├── associations.py           # Association / tracking helpers
├── tracker.py                # Tracker logic
├── utils.py                  # Loss, encoding, NMS, evaluation (confusion matrix, mAP)
├── compute_ap.py             # AP computation (used by utils)
├── new_new_graph_tracking/   # VIS inference & submission (mainVIS_Track4.py, bipartite matching)
├── scripts/                  # Standalone utilities (see scripts/README.md)
│   ├── overlay_vis_on_video.py   # Overlay VIS masks on video (from JSON or dataset)
│   ├── concat_vis_videos.py      # Concatenate vis_v*.mp4 into one video
│   ├── shrink_vis_video.py      # Trim combined video under 100 MB for GitHub
│   └── evaluate.py              # YouTube-VOS evaluation (pycocotools)
├── assets/                   # Demo media (README video, figures)
│   └── vis_results_combined_under100mb.mp4
├── info/                     # Anchor files (e.g. anchors_yt_16.txt)
├── papers/                   # Reference PDFs (YOLOV4, PANet, MOTS, etc.)
├── old/                      # Legacy code (previous tracking, loaders)
├── VIS_10_Rossetti.pdf       # Challenge technical report
└── README.md
```

---

## References

- **YOLOV4**: Bochkovskiy et al., [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934).
- **YouTube-VIS**: Yang et al., [YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark](https://youtube-vos.org/dataset/).
- **PANet**: Liu et al., [Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534).
- **Automatic Loss Balancing**: Kendall et al., [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115).
- **Challenge**: [3rd Large-scale Video Object Segmentation Challenge (CVPR 2021)](https://youtube-vos.org/challenge/2021/leaderboard/) — Track 2: Video Instance Segmentation.

---

## Authors & license

**Authors**: Simone Rossetti, Temirlan Zharbynbek, Fiora Pirri — Alcor LAB, Sapienza Università di Roma.

**License**: [MIT](LICENSE).

If you use this code or the technical report, please cite the [YouTube-VOS 2021 challenge](https://youtube-vos.org/challenge/2021/leaderboard/) and our technical report (VIS_10_Rossetti.pdf).
