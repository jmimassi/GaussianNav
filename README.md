<p align="center">
  <img src="media/logo.png" alt="GaussianNav Logo" width="350">
</p>

<h1 align="center">GaussianNav</h1>


<p align="center">
  <b>Safe Camera Path Planning & Rendering in 3D Gaussian Splatting Scenes</b>
</p>


<p align="center">
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Demo-yellow" alt="Demo"></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-Nextcloud-0082c9?logo=nextcloud&logoColor=white" alt="Dataset"></a>
</p>

---

## Overview

**GaussianNav** generates smooth, safe camera trajectories through 3D Gaussian Splatting (3DGS) reconstructions. It combines frustum-based volume carving, A\* pathfinding, and coverage-aware rotation correction to produce high-quality renderings with dense optical flow.

## Features

- **Frustum Carving** — Computes a navigable volume from training camera depth maps
- **A\* Pathfinding** — Plans paths through a density-weighted navigation graph with random waypoints
- **Coverage Correction** — Backward-propagation algorithm that adjusts camera rotations to avoid empty regions and oversized splats
- **Optical Flow** — Generates geometrically consistent flow maps in KITTI format
- **3D Visualization** — Outputs interactive plots of the planned path overlaid on the point cloud

## Prerequisites

- **CUDA** >= 12.x
- **COLMAP** — required to extract camera poses from images before training ([install guide](https://colmap.github.io/install.html))
- **Python** >= 3.10

## Installation

```bash
# Clone submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn

# Install dependencies
pip install -r requirements.txt

# Build CUDA submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Usage

### Option A — You don't have a trained model yet

First, run COLMAP on your images to extract camera poses, then configure `create_splat.py` and run it:

```bash
python create_splat.py
```

Then configure `gaussiannav.py` with the trained model path and run:

```bash
python gaussiannav.py
```

### Option B — You already have a trained model (.ply)

If you already have a trained 3DGS model (with `point_cloud/`, `cameras.json`, `cfg_args`), skip training. Configure `gaussiannav.py` with your model path and run:

```bash
python gaussiannav.py
```

All parameters are configured directly in the Python files. Edit the `args` section at the top of each script before running.

## Dataset

Place your data under `dataset/`:

```
dataset/
├── ssd1/
├── ssd2/
└── ssd3/
```

## Output Structure

```
rendered/
├── img/                 # RGB frames with coverage overlay
├── flow/                # Optical flow (KITTI format)
├── flow_vis/            # Flow visualizations
├── coverage_masks/      # Per-frame coverage maps
├── coverage.txt         # Per-frame metrics (empty %, splat scale)
└── path_visualization.png
```

## Acknowledgements

Our code is inspired by the work of the authors of [UnifiedGeneralization](https://github.com/HanLingsgjk/UnifiedGeneralization) — check it out!
