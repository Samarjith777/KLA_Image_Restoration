# KLA AI Hackathon - AI-Based Image Restoration

## 1. Project Overview

This project addresses image restoration for KLA semiconductor inspection data.

The objective is to reconstruct high-resolution 256x256 images from degraded 128x128 input images while preserving important structural and fine-detail information.

The final submission uses the **V4 Noise-Aware Detail Restoration** model.

---

## 2. Final Model

### V4 Noise-Aware Detail Restoration

The V4 model combines:

- Pixel-level reconstruction
- Edge-aware restoration
- Fine-detail preservation
- Noise-aware processing

### Model Configuration

```text
Model:           V4NoiseAwareRestoration
Input channels:  1
Output channels: 1
Channels:        64
Blocks:          6
Input size:      128x128
Output size:     256x256