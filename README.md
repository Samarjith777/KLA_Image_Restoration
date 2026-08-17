# KLA Image Restoration — V4 Noise-Aware Detail Restoration

**AI-Powered Super-Resolution for Semiconductor Wafer Inspection Images**

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.13.0+cu126-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

---

## Overview

**KLA Image Restoration** is an end-to-end deep learning pipeline that reconstructs high-fidelity 256×256 semiconductor inspection images from degraded 128×128 inputs. Built for the **Semicon 2026 Hackathon**, the project addresses a critical challenge in semiconductor manufacturing: preserving nanoscale structural detail while suppressing noise amplification during 2× super-resolution.

Unlike generic upscaling methods that blur fine features or amplify artifacts, the **V4 Noise-Aware Detail Restoration Model** uses a **bicubic residual learning** strategy combined with a learned noise-gating mechanism. The model predicts a residual correction on top of a bicubic baseline, allowing the network to focus exclusively on restoring genuine fine detail while a dedicated noise estimator suppresses artifact generation in noisy regions.

### Key Capabilities

- **2× Super-Resolution**: 128×128 → 256×256 with structural fidelity preservation
- **Noise-Aware Gating**: Learned noise map `n ∈ [0,1]` selectively suppresses detail restoration in noisy regions
- **Edge & Detail Preservation**: Dual-scale detail extraction (local 3×3 + dilated 3×3) with Laplacian detail loss
- **Lightweight Architecture**: ~800K parameters — deployable on edge inspection hardware
- **Standalone Evaluator**: Zero-config inference script for production benchmarking

---

## Verified Benchmarks & Metrics

Metrics computed on 320 paired validation samples (local project validation — not official KLA H100 benchmark scores):

| Metric | V4 Model | Bicubic Baseline | Improvement |
|--------|----------|------------------|-------------|
| **PSNR** | **28.3314 dB** | ~26.0 dB | **+2.3 dB** |
| **SSIM** | **0.765930** | ~0.65 | **+17.8%** |
| **LPIPS** | **0.258063** | ~0.38 | **-32.1%** |
| **Parameters** | ~800K | — | Lightweight |
| **Inference Time** | ~12 ms/image (RTX 3050) | — | Real-time capable |

> **Note:** Official KLA benchmark evaluation should be performed by the evaluation team using the submitted standalone evaluator and designated H100 hardware/environment.

---

## Architecture & Model Pipeline

```
flowchart LR
    classDef inputNode fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc;
    classDef procNode fill:#2e1065,stroke:#c084fc,stroke-width:1.5px,color:#f8fafc;
    classDef outputNode fill:#064e3b,stroke:#34d399,stroke-width:1.5px,color:#f8fafc;

    A[(128×128 Degraded<br/>.npy Input)]:::inputNode --> B[Bicubic Baseline<br/>2× Upsample]:::procNode
    A --> C[Shallow Feature Head<br/>1→64 ch]:::procNode
    C --> D[Residual Body<br/>6× ResBlock]:::procNode
    D --> E[Detail Branch<br/>Local + Dilated]:::procNode
    D --> F[Noise Estimator<br/>Sigmoid Map]:::procNode
    E --> G[Noise-Aware Gating<br/>detail × (1 − noise)]:::procNode
    D --> G
    G --> H[Feature Fusion<br/>Cat + Conv]:::procNode
    H --> I[PixelShuffle ×2<br/>Upsample]:::procNode
    I --> J[Residual Head<br/>1 ch]:::procNode
    B --> K[Output<br/>base + α·residual]:::outputNode
    J --> K
    K --> L[(256×256 Restored<br/>.npy Output)]:::outputNode
```

### Pipeline Stages

1. **Bicubic Baseline** — Fast 2× upsampling preserves global structure and provides a stable reconstruction anchor
2. **Shallow Feature Extraction** — 1→64 channel projection captures low-level features
3. **Residual Body (6 Blocks)** — Deep residual learning extracts structural features without gradient degradation
4. **Dual-Scale Detail Branch** — Parallel 3×3 local and 3×3 dilated convolutions capture fine detail at multiple receptive fields
5. **Noise Estimation** — Lightweight CNN predicts per-pixel noise probability `n ∈ [0,1]`
6. **Noise-Aware Gating** — Detail features are gated by `(1 − noise_map)`, suppressing artifact amplification in noisy regions
7. **Feature Fusion & Upsampling** — Concatenated structure + gated detail features fused via 1×1/3×3 convolutions, upsampled via PixelShuffle
8. **Residual Reconstruction** — Final residual scaled by learnable `α = 0.1` and added to bicubic baseline

---

## The V4 Model Components

| Component | Purpose | Key Design |
|-----------|---------|------------|
| **ResidualBlock** | Deep feature extraction | 2× Conv3×3 + ReLU with skip connection |
| **DetailBlock** | Multi-scale detail capture | Local 3×3 + Dilated 3×3 fusion |
| **NoiseEstimator** | Per-pixel noise detection | 3-layer CNN → Sigmoid, outputs `n ∈ [0,1]` |
| **DetailEstimator** | Fine-detail feature enhancement | 3-layer CNN refines detail representation |
| **PixelShuffle** | Efficient sub-pixel upsampling | Conv to `channels×4` → PixelShuffle(2) |
| **Residual Scaling** | Stabilized residual learning | Learnable `α = 0.1` prevents instability |

### Loss Function

The V4 training objective combines three complementary terms:

```
L_total = 1.0 × L_pixel + 0.15 × L_edge + 0.03 × L_detail
```

- **Pixel Loss (L1)**: Direct reconstruction accuracy — `L1(pred, target)`
- **Edge Loss (L1)**: Structural preservation via Sobel edge maps — `L1(Sobel(pred), Sobel(target))`
- **Detail Loss (L1)**: Fine-detail fidelity via Laplacian maps — `L1(Laplacian(pred), Laplacian(target))`

---

## Comparison

| Approach | Noise Handling | Detail Preservation | Parameters | Inference Speed |
|----------|---------------|---------------------|------------|-----------------|
| **V4 (Ours)** | ✅ Learned noise gating | ✅ Dual-scale detail | ~800K | ⚡ Real-time |
| Bicubic Interpolation | ❌ Amplifies noise | ❌ Blurs detail | — | ⚡⚡⚡ Instant |
| Simple CNN (SRCNN-like) | ⚠️ Mild suppression | ⚠️ Limited detail | ~50K | ⚡⚡⚡ Fast |
| Heavy U-Net | ⚠️ Aggressive smoothing | ✅ Good detail | ~30M | 🐢 Slow |
| SwinIR / Transformer | ✅ Good | ✅ Excellent | ~10M | 🐢🐢 Very Slow |

**V4 strikes the optimal balance** for industrial semiconductor inspection: lightweight enough for edge deployment, yet sophisticated enough to outperform bicubic and simple CNN baselines by a significant margin.

---

## Repository Structure

```
KLA_Image_Restoration/
├── checkpoint/
│   └── v4_noise_detail_best.pth       # Trained V4 model weights
├── models/
│   └── v4_noise_detail.py             # V4 architecture implementation
├── outputs/
│   └── *.npy                          # 3,200 restored test outputs (256×256)
├── results/
│   └── training_history.txt           # Full training log & metrics
├── presentation/
│   └── ...                            # Project presentation materials
├── evaluate.py                        # Standalone inference script
├── train_v4.py                        # V4 training / reproduction script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.6 support (RTX 3050 or higher recommended)
- Git LFS (for downloading model checkpoints)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Samarjith777/KLA_Image_Restoration.git
cd KLA_Image_Restoration

# 2. Pull Git LFS files (model weights)
git lfs install
git lfs pull

# 3. Create virtual environment
python -m venv .venv

# 4. Install PyTorch with CUDA 12.6
.venv\Scripts\python.exe -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 5. Install remaining dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt --no-deps
```

### Standalone Evaluation

Run inference on a directory of degraded 128×128 `.npy` images:

```bash
.venv\Scripts\python.exe evaluate.py <input_directory> <output_directory>
```

Example:
```bash
.venv\Scripts\python.exe evaluate.py .\test_images .\outputs_test
```

The evaluator will:
- Load the trained V4 checkpoint automatically
- Process all `.npy` files in the input directory
- Output 256×256 float32 `.npy` files clamped to `[0, 1]`
- Preserve original filenames
- Verify input/output counts match

### Training (Optional)

To reproduce or fine-tune the V4 model:

```bash
.venv\Scripts\python.exe train_v4.py
```

Training configuration:
- **Epochs:** 30
- **Batch Size:** 4
- **Learning Rate:** 1e-4 (AdamW)
- **Scheduler:** Cosine Annealing (T_max=30, eta_min=1e-7)
- **Loss Weights:** Pixel=1.0, Edge=0.15, Detail=0.03

---

## Input / Output Specification

| | Input | Output |
|---|---|---|
| **Format** | `.npy` (NumPy array) | `.npy` (NumPy array) |
| **Shape** | `(128, 128)` | `(256, 256)` |
| **Channels** | 1 (Grayscale) | 1 (Grayscale) |
| **Dtype** | `float32` | `float32` |
| **Value Range** | `[0, 1]` | `[0, 1]` (clamped) |
| **Test Set Size** | 3,200 images | 3,200 images |

---

## Hardware & Performance

| | Specification |
|---|---|
| **Development GPU** | NVIDIA GeForce RTX 3050 Laptop GPU |
| **CUDA Version** | 12.6 |
| **PyTorch Version** | 2.13.0+cu126 |
| **Inference Speed** | ~12 ms/image (RTX 3050) |
| **Model Size** | ~800K parameters (~3.2 MB `.pth`) |
| **Official Benchmark** | NVIDIA H100 (evaluation team hardware) |

---

## Hackathon Submission Details

### Submission Information

- **Hackathon:** Semicon 2026
- **Track:** AI for Semiconductor Manufacturing
- **Challenge:** KLA Image Restoration — Reconstruct 256×256 images from degraded 128×128 inputs
- **Framework:** PyTorch 2.13
- **Model:** V4 Noise-Aware Detail Restoration
- **Repository:** https://github.com/Samarjith777/KLA_Image_Restoration

### Rubric Alignment

| Evaluation Criteria | How V4 Delivers |
|---|---|
| **Reconstruction Quality** | PSNR 28.33 dB, SSIM 0.766, LPIPS 0.258 on 320 validation pairs |
| **Detail Preservation** | Dual-scale detail branch + Laplacian detail loss maintain fine structural features |
| **Noise Robustness** | Learned noise map gates detail restoration, preventing artifact amplification |
| **Efficiency & Deployability** | ~800K parameters, ~12ms inference on RTX 3050 — edge-deployment ready |
| **Reproducibility** | Standalone `evaluate.py` with zero manual config; training script included |
| **Submission Validity** | 3,200 output files verified: correct shape (256×256), dtype (float32), range ([0,1]) |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Semicon 2026 — KLA Image Restoration</b><br/>
  <sub>V4 Noise-Aware Detail Restoration Model</sub>
</p>
