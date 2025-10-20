# EzPIc Face Recognition Pipeline# 📸 EzPIc Face Recognition Pipeline# 📸 EzPIc Face Recognition Pipeline📸 EzPIc Face Recognition Pipeline (Hybrid Architecture)



High-accuracy, real-time face recognition system with pose invariance for intelligent photo retrieval.



## Overview**Hybrid/On-Device Architecture for High-Accuracy Real-Time Face Recognition**Repository Link: https://github.com/ay-ush-17/Ezpic_Pipeline



This pipeline achieves **95.9% accuracy** on cross-pose face recognition while maintaining **~34ms inference speed**. It solves the pose-invariance challenge through a combination of robust deep learning models and multi-view enrollment.



## FeaturesRepository: [ay-ush-17/Ezpic_Pipeline](https://github.com/ay-ush-17/Ezpic_Pipeline)**Hybrid/On-Device Architecture for High-Accuracy Real-Time Face Recognition**



- ✅ **95.9% Accuracy** on LFW cross-pose subset

- ✅ **~34ms Inference** on target hardware  

- ✅ **Pose Invariant** through multi-view enrollment (frontal, left, right profiles)## 🎯 Project StatusThis project implements the core recognition engine for the EzPIc Photo Retrieval System. It is designed around a Hybrid/On-Device architecture to solve the trade-off between high pose-invariant accuracy and sub-100ms real-time processing speed, fulfilling all core project mandates.

- ✅ **Production Ready** with validated Float32 ONNX model



## Architecture

**Functionally Complete (Accuracy Confirmed)**Repository: [ay-ush-17/Ezpic_Pipeline](https://github.com/ay-ush-17/Ezpic_Pipeline)

### Pipeline Flow



1. **Detection** → YuNet detects faces and extracts 5 facial landmarks

2. **Alignment** → Custom affine transformation normalizes face to 112×112The pipeline successfully runs the most robust verification models available and is ready for final optimization for deployment.🎯 Project Status: Functionally Complete (Accuracy Confirmed)

3. **Embedding** → ArcFace-ResNet50 generates 512D face embedding

4. **Matching** → Cosine similarity against enrolled templates



### Pose Invariance### Mandate Resolution## OverviewThe pipeline successfully runs the most robust verification models available and is ready for final optimization for deployment.



The system handles pose variation through:



- **Model-Level**: ArcFace-ResNet50 trained on diverse poses| Mandate | Implementation | Status |

- **Data-Level**: Multi-view enrollment captures 3 templates per person (frontal, left, right)

|---------|----------------|--------|

## Installation

| **Accuracy (95%+ LFW)** | ArcFace-ResNet50 + Custom Alignment | ✅ SOLVED (95.9% accuracy on cross-pose subset) |This project implements the core recognition engine for the EzPIc Photo Retrieval System. It uses a Hybrid/On-Device architecture that balances high pose-invariant accuracy with sub-100ms real-time processing speed, fulfilling all core project mandates.| Mandate | Final Implementation | Viability Status |

### Requirements

| **Pose Invariance** | Multi-View Enrollment (3 Poses) | ✅ SOLVED (Fixed side-profile recognition failure) |

- Python 3.8+

- Dependencies: `onnxruntime`, `opencv-python`, `numpy`, `scikit-learn`, `pillow`| **Speed (<100ms)** | Float ONNX Model (166 MB) | ✅ SOLVED (~34 ms on target hardware) || Accuracy (95%+ LFW) | ArcFace-ResNet50 + Custom Alignment | ✅ SOLVED. Achieved 95.9% accuracy on cross-pose subset. |



### Setup| **Deployment Asset** | PTQ/QDQ INT8 | ⚠️ BROKEN (Quantization caused accuracy flaw: 95.2%→67.4%) |



1. **Clone repository**## 🎯 Project Status| Pose Invariance | Multi-View Enrollment (3 Poses) | ✅ SOLVED. Fixed the critical side-profile recognition failure. |

```bash

git clone https://github.com/ay-ush-17/Ezpic_Pipeline## 📝 Overview

cd Ezpic_Pipeline

```| Speed (<100ms) | Float ONNX Model (166 MB) | ✅ SOLVED. Meets speed mandate on target hardware (∼34 ms). |



2. **Install dependencies**This project implements the core recognition engine for the EzPIc Photo Retrieval System. It uses a Hybrid/On-Device architecture that balances high pose-invariant accuracy with sub-100ms real-time processing speed, fulfilling all core project mandates.

```bash

pip install onnxruntime opencv-python numpy scikit-learn pillow**Functionally Complete (Accuracy Confirmed)**| Deployment Asset | PTQ/QDQ INT8 | ⚠️ BROKEN. Quantization created an accuracy flaw (95.2%→67.4%) and is the final, open technical hurdle. |

```

## ⚙️ Architecture Overview

3. **Download models** and place in `Models/` directory:

   - YuNet: `face_detection_yunet_2023mar_int8.onnx`

   - ArcFace: `w600k_r50.onnx`

The system uses a unified pipeline across detection and recognition for maximum robustness.

## Usage

The pipeline successfully runs the most robust verification models available and is ready for final optimization for deployment.⚙️ Final Architecture Overview

### LFW Benchmark

### Core Pipeline Flow

Test against industry-standard Labeled Faces in the Wild dataset:

The system uses a unified pipeline across detection and recognition for maximum robustness:

```bash

python scripts/lfw_benchmark.py1. **Fast Detection**: YuNet (.onnx) locates the face and extracts 5 landmarks

```

2. **Alignment**: Custom Affine Transformation (OpenCV) aligns and crops face to 112×112 input size### Mandate Resolution

### Interactive GUI

3. **Recognition**: ArcFace-ResNet50 (w600k_r50.onnx) generates 512D vector (Face Template)

Test enrollment and matching with camera:

1. The Core Pipeline Flow

```bash

python whole_face_scan_gui.py### High-Accuracy Strategy

```

| Mandate | Implementation | Status |Fast Detection: YuNet (.onnx) locates the face and 5 landmarks.

## Performance

The project solves the 2D → 3D pose problem using two methods simultaneously:

| Metric | Target | Achieved | Status |

|--------|--------|----------|--------||---------|----------------|--------|

| Accuracy (LFW) | ≥95% | 95.9% | ✅ |

| Inference Speed | <100ms | ~34ms | ✅ |- **Model Fix**: Deep ArcFace-ResNet50 model with intrinsic robustness to pose variation

| Pose Invariance | Side profiles | Multi-view solved | ✅ |

- **Data Fix**: Multi-View Enrollment logic capturing three templates (Frontal, Left, Right) per user, guaranteeing a match against any angle| **Accuracy (95%+ LFW)** | ArcFace-ResNet50 + Custom Alignment | ✅ SOLVED (95.9% accuracy on cross-pose subset) |Alignment: Custom Affine Transformation (OpenCV Math) uses landmarks to align and crop the face to the perfect 112×112 input size.

## Known Issues



**INT8 Quantization**: Attempted PTQ/QDQ INT8 quantization caused severe accuracy drop (95.2% → 67.4%). Currently using Float32 ONNX model (166MB) for deployment.

## 🛠️ Setup and Installation| **Pose Invariance** | Multi-View Enrollment (3 Poses) | ✅ SOLVED (Fixed side-profile recognition failure) |

## Status



**Functionally Complete** - Core pipeline validated and deployment-ready. Model compression without accuracy loss remains an open challenge.

### Prerequisites| **Speed (<100ms)** | Float ONNX Model (166 MB) | ✅ SOLVED (~34 ms on target hardware) |Recognition: ArcFace-ResNet50 (w600k_r50.onnx) generates the final 512D vector (the Face Template).



- Python 3.8+| **Deployment Asset** | PTQ/QDQ INT8 | ⚠️ BROKEN (Quantization caused accuracy flaw: 95.2%→67.4%) |

- External Libraries: numpy, opencv-python, onnxruntime, scikit-learn

2. High-Accuracy Strategy

### External Models (Download Required)

## ⚙️ Architecture OverviewThe project solves the 2D → 3D pose problem by using two methods simultaneously:

- **YuNet Detector**: `face_detection_yunet_2023mar_int8.onnx`

- **ArcFace Recognizer**: `w600k_r50.onnx` (Float Model - Validated)



### Installation StepsThe system uses a unified pipeline across detection and recognition for maximum robustness.Model Fix: Using the deep ArcFace-ResNet50 model, which is intrinsically robust to pose variation.



1. **Clone Repository**

   ```bash

   git clone https://github.com/ay-ush-17/Ezpic_Pipeline### Core Pipeline FlowData Fix: Implementing Multi-View Enrollment logic, which captures and stores three templates (Frontal, Left, Right) per user, guaranteeing a match against any angle in the user's photo library.

   cd Ezpic_Pipeline

   ```



2. **Install Dependencies**1. **Fast Detection**: YuNet (.onnx) locates the face and extracts 5 landmarks🛠️ Setup and Execution

   ```bash

   python -m pip install onnxruntime opencv-python numpy scikit-learn pillow2. **Alignment**: Custom Affine Transformation (OpenCV) aligns and crops face to 112×112 input sizePrerequisites

   ```

3. **Recognition**: ArcFace-ResNet50 (w600k_r50.onnx) generates 512D vector (Face Template)Python 3.8+ (must be the same version used for development)

3. **Place Models**

   - Create a `Models/` directory

   - Place `w600k_r50.onnx` and `face_detection_yunet_2023mar_int8.onnx` inside

### High-Accuracy StrategyExternal Libraries: numpy, opencv-python, onnxruntime, scikit-learn

## 🚀 Running the Pipeline



### Verification Benchmark

The project solves the 2D → 3D pose problem using two methods simultaneously:External Models (Download required):

Test model performance against the industry-standard LFW Cross-Pose protocol:



```bash

python scripts/lfw_benchmark.py- **Model Fix**: Deep ArcFace-ResNet50 model with intrinsic robustness to pose variationYuNet Detector: face_detection_yunet_2023mar_int8.onnx

```

- **Data Fix**: Multi-View Enrollment logic capturing three templates (Frontal, Left, Right) per user, guaranteeing a match against any angle

### Multi-View GUI

ArcFace Recognizer: w600k_r50.onnx (The validated Float Model)

Interactive testing of Enrollment (3 poses) and Matching logic:

## 🛠️ Setup and Installation

```bash

python whole_face_scan_gui.pyInstallation

```

### PrerequisitesClone Repository:

## ⚠️ Deployment Recommendations



Due to fatal accuracy loss during INT8 quantization, the recommended deployment asset is the **166 MB Float ONNX Model** (`w600k_r50.onnx`). It is the only model that guarantees the required 95% accuracy score.

- Python 3.8+git clone [https://github.com/ay-ush-17/Ezpic_Pipeline](https://github.com/ay-ush-17/Ezpic_Pipeline)

- External Libraries: numpy, opencv-python, onnxruntime, scikit-learncd Ezpic_Pipeline



### External Models (Download Required)

Install Dependencies:

- **YuNet Detector**: `face_detection_yunet_2023mar_int8.onnx`

- **ArcFace Recognizer**: `w600k_r50.onnx` (Float Model - Validated)# Use your environment's Python executable path if 'pip' is not recognized

python -m pip install onnxruntime opencv-python numpy scikit-learn pillow

### Installation Steps



1. **Clone Repository**Place Models: Create a Models/ directory and place the downloaded w600k_r50.onnx and face_detection_yunet_2023mar_int8.onnx files inside it.

   ```bash

   git clone https://github.com/ay-ush-17/Ezpic_PipelineRunning the Verification Benchmark

   cd Ezpic_PipelineUse the final benchmark script to test the model's performance against the industry-standard LFW Cross-Pose protocol.

   ```

# Set your PYTHONPATH to include the WORKING CODE folder if necessary

2. **Install Dependencies**# Example: python scripts/lfw_benchmark.py --max-images 2000

   ```bashpython scripts/lfw_benchmark.py

   python -m pip install onnxruntime opencv-python numpy scikit-learn pillow

   ```

Running the Multi-View GUI

3. **Place Models**The GUI allows interactive testing of the Enrollment (3 poses) and Matching (Minimum Distance) logic.

   - Create a `Models/` directory

   - Place `w600k_r50.onnx` and `face_detection_yunet_2023mar_int8.onnx` insidepython whole_face_scan_gui.py



## 🚀 Running the Pipeline

⚠️ Final Deployment Note

### Verification BenchmarkDue to the fatal accuracy loss during the INT8 quantization attempt, the recommended deployment asset is the 166 MB Float ONNX Model (w600k_r50.onnx), as it is the only model that guarantees the required 95% accuracy score.


Test model performance against the industry-standard LFW Cross-Pose protocol:

```bash
python scripts/lfw_benchmark.py
```

### Multi-View GUI

Interactive testing of Enrollment (3 poses) and Matching logic:

```bash
python whole_face_scan_gui.py
```

## ⚠️ Deployment Recommendations

Due to fatal accuracy loss during INT8 quantization, the recommended deployment asset is the **166 MB Float ONNX Model** (`w600k_r50.onnx`). It is the only model that guarantees the required 95% accuracy score.
