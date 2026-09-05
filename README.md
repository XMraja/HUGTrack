# Geometry-Language Distilled Hull-Aware Convolution for Real-Time UAV Multi-Vessel Tracking 

**Zijie Zhang<sup>1</sup>, Changhong Fu<sup>1,†</sup>, Mengyuan Li<sup>1</sup>, Yongkang Cao<sup>1</sup>, Haobo Zuo<sup>2</sup>, Guangze Zheng<sup>2</sup>, Bowen Li<sup>3</sup>**

<sup>1</sup>Tongji University, Shanghai, China  
<sup>2</sup>The University of Hong Kong, Hong Kong, China  
<sup>3</sup>Carnegie Mellon University, Pittsburgh, USA  
<sup>†</sup>Corresponding author



## Abstract

Visual multi-vessel tracking is a key perception capability for intelligent transportation systems in waterways, supporting maritime traffic management and waterway surveillance. However, vessels exhibit slender structures, weak textures, and high inter-instance similarity, requiring subtle contour cues and temporally consistent geometry to be preserved under large-scale variations and viewpoint perturbations induced by unmanned aerial vehicles (UAVs). On resource-constrained UAV-mounted electro-optical devices, this further calls for geometrically expressive, structurally controllable, and computationally efficient representations, which current methods struggle to maintain. To this end, we propose HUGTrack, a visual multi-vessel tracking framework for real-time deployment on UAV platforms. Geometry-language distilled hull-aware convolution (GLD-HConv) learns coarse-to-fine hull representations through structured coordinate alignment, bounded instance adaptation, and geometry-language supervision. Uncertainty-calibrated motion-geometry association (UCMGA) maintains stable trajectories under uncertain observations, irregular vessel motion, and UAV-induced viewpoint changes. We further establish GeoVesselMOT, a large-scale UAV-captured multi-vessel tracking benchmark with trajectory labels, instance masks, and geometry-language annotations.

---

## 🔥 Highlights

- **GLD-HConv** learns structured and instance-adaptive vessel representations with training-only geometry-language supervision.
- **UCMGA** maintains stable trajectories through uncertainty-calibrated motion and geometry association.
- **GeoVesselMOT** provides trajectory labels, instance masks, geometry records, and geometry-language descriptions.
- **Edge-ready inference** removes the training-only language and auxiliary branches during testing.

## 🏗️ Repository Structure

```text
├── infer.py                              # Detection test entry
├── track.py                              # UCMGA tracking entry
├── evaluate.py                           # Tracking evaluation entry
├── requirements.txt
├── weights/
│   └── geovesselmot.pt                   # Released detector checkpoint
├── tracker/
│   └── geovesselmot-params.json          # Per-sequence tracking parameters
├── cam_para/geovesselmot/test/           # Camera parameters
├── dmc/geovesselmot/test/                # Viewpoint deltas
├── detector/                             # Detection-to-ground-plane mapping
├── eval/                                 # GeoVesselMOT evaluation
└── ultralytics/                          # Detector inference pipeline
```

---

## 🚀 Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.13
- CUDA-compatible GPU for accelerated inference

### Setup Environment

```bash
conda create -n hugtrack python=3.10 -y
conda activate hugtrack
pip install -r requirements.txt
```

## 📦 Dataset Preparation

### GeoVesselMOT Dataset

Arrange the GeoVesselMOT sequences as follows. Sequence names must use the `geovesselmot-XX` form.

```text
geovesselmot/
├── test/
│   ├── geovesselmot-05/
│   │   ├── img1/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── gt/
│   │       └── gt.txt
│   └── ...
├── train/
├── val/
└── seqmaps/
    └── geovesselmot-test.txt
```

---

## 📊 Test

Run detection:

```bash
python infer.py \
    --data-root /path/to/geovesselmot/test \
    --out-root outputs/detections
```

Run tracking with the released sequence-specific parameters:

```bash
python track.py \
    --detections outputs/detections \
    --output-root outputs/tracks \
    --run-name hugtrack
```

Evaluate the tracking results:

```bash
python evaluate.py \
    --gt-root /path/to/geovesselmot/test \
    --track-root outputs/tracks \
    --seqmap /path/to/geovesselmot/seqmaps/geovesselmot-test.txt \
    --run-name hugtrack
```

Key test arguments:

| Argument | Default | Description |
|---|---|---|
| `infer.py --weights` | `weights/geovesselmot.pt` | Released GLD-HConv checkpoint |
| `infer.py --imgsz` | `320` | Input resolution |
| `infer.py --conf` | `0.25` | Detection confidence threshold |
| `track.py --params` | `tracker/geovesselmot-params.json` | Per-sequence tracking parameters |
| `track.py --camera-dir` | `cam_para/geovesselmot/test` | Camera parameter directory |
| `track.py --viewpoint-dir` | `dmc/geovesselmot/test` | Viewpoint-delta directory |

### Test Results

| Method | HOTA | MOTA | IDF1 | IDs | DetA | DetRe | DetPr | AssA | AssRe | AssPr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HUGTrack | 68.8444 | 56.0986 | 67.4255 | 999.0000 | 56.9033 | 58.3777 | 96.2005 | 86.5893 | 87.5113 | 96.9107 |

---

## 🏋️ Training

HUGTrack follows the tracking-by-detection paradigm. Training applies geometry-language distillation to the **GLD-HConv** detector, while **UCMGA** is training-free and operates on the detector outputs during testing.

The detector training data use the Ultralytics format:

```text
geovesselmot_detection/
├── images/{train,val,test}/<sequence>/<frame>.jpg
└── labels/{train,val,test}/<sequence>/<frame>.txt
```

Each label follows the normalized YOLO format `<class> <cx> <cy> <w> <h>`. Geometry-language distillation additionally uses the following training targets:

```text
train_gld_image_targets.jsonl
train_instances_with_prompts.jsonl
text_bank.npy
```

This repository is the test-only release and therefore does not include the training entry, dataset construction utilities, or geometry-language target generation scripts.

---

## 📄 License

This project is licensed under the AGPL-3.0 License.
