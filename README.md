# EzPIc Face Recognition Pipeline

High-accuracy, real-time face recognition system with pose invariance for intelligent photo retrieval.

## Overview

This pipeline achieves **95.9% accuracy** on cross-pose face recognition while maintaining **~34ms inference speed**. It solves the pose-invariance challenge through a combination of robust deep learning models and multi-view enrollment.

## Features

- ✅ **95.9% Accuracy** on LFW cross-pose subset
- ✅ **~34ms Inference** on target hardware  
- ✅ **Pose Invariant** through multi-view enrollment (frontal, left, right profiles)
- ✅ **Production Ready** with validated Float32 ONNX model

## Architecture

### Pipeline Flow

1. **Detection** → YuNet detects faces and extracts 5 facial landmarks
2. **Alignment** → Custom affine transformation normalizes face to 112×112
3. **Embedding** → ArcFace-ResNet50 generates 512D face embedding
4. **Matching** → Cosine similarity against enrolled templates

### Pose Invariance

The system handles pose variation through:

- **Model-Level**: ArcFace-ResNet50 trained on diverse poses
- **Data-Level**: Multi-view enrollment captures 3 templates per person (frontal, left, right)

## Installation

### Requirements

- Python 3.8+
- Dependencies: `onnxruntime`, `opencv-python`, `numpy`, `scikit-learn`, `pillow`

### Setup

1. **Clone repository**
```bash
git clone https://github.com/ay-ush-17/Ezpic_Pipeline
cd Ezpic_Pipeline
```

2. **Install dependencies**
```bash
pip install onnxruntime opencv-python numpy scikit-learn pillow
```

3. **Download models** and place in `Models/` directory:
   - YuNet: `face_detection_yunet_2023mar_int8.onnx`
   - ArcFace: `w600k_r50.onnx`

## Usage

### LFW Benchmark

Test against industry-standard Labeled Faces in the Wild dataset:

```bash
python scripts/lfw_benchmark.py
```

### Interactive GUI

Test enrollment and matching with camera:

```bash
python whole_face_scan_gui.py
```

## Performance

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Accuracy (LFW) | ≥95% | 95.9% | ✅ |
| Inference Speed | <100ms | ~34ms | ✅ |
| Pose Invariance | Side profiles | Multi-view solved | ✅ |

## Known Issues

**INT8 Quantization**: Attempted PTQ/QDQ INT8 quantization caused severe accuracy drop (95.2% → 67.4%). Currently using Float32 ONNX model (166MB) for deployment.

## Status

**Functionally Complete** - Core pipeline validated and deployment-ready. Model compression without accuracy loss remains an open challenge.
