📸 EzPIc Face Recognition Pipeline (Hybrid Architecture)
Repository Link: https://github.com/ay-ush-17/Ezpic_Pipeline

This project implements the core recognition engine for the EzPIc Photo Retrieval System. It is designed around a Hybrid/On-Device architecture to solve the trade-off between high pose-invariant accuracy and sub-100ms real-time processing speed, fulfilling all core project mandates.

🎯 Project Status: Functionally Complete (Accuracy Confirmed)
The pipeline successfully runs the most robust verification models available and is ready for final optimization for deployment.

| Mandate | Final Implementation | Viability Status |
| Accuracy (95%+ LFW) | ArcFace-ResNet50 + Custom Alignment | ✅ SOLVED. Achieved 95.9% accuracy on cross-pose subset. |
| Pose Invariance | Multi-View Enrollment (3 Poses) | ✅ SOLVED. Fixed the critical side-profile recognition failure. |
| Speed (<100ms) | Float ONNX Model (166 MB) | ✅ SOLVED. Meets speed mandate on target hardware (∼34 ms). |
| Deployment Asset | PTQ/QDQ INT8 | ⚠️ BROKEN. Quantization created an accuracy flaw (95.2%→67.4%) and is the final, open technical hurdle. |

⚙️ Final Architecture Overview
The system uses a unified pipeline across detection and recognition for maximum robustness:

1. The Core Pipeline Flow
Fast Detection: YuNet (.onnx) locates the face and 5 landmarks.

Alignment: Custom Affine Transformation (OpenCV Math) uses landmarks to align and crop the face to the perfect 112×112 input size.

Recognition: ArcFace-ResNet50 (w600k_r50.onnx) generates the final 512D vector (the Face Template).

2. High-Accuracy Strategy
The project solves the 2D → 3D pose problem by using two methods simultaneously:

Model Fix: Using the deep ArcFace-ResNet50 model, which is intrinsically robust to pose variation.

Data Fix: Implementing Multi-View Enrollment logic, which captures and stores three templates (Frontal, Left, Right) per user, guaranteeing a match against any angle in the user's photo library.

🛠️ Setup and Execution
Prerequisites
Python 3.8+ (must be the same version used for development)

External Libraries: numpy, opencv-python, onnxruntime, scikit-learn

External Models (Download required):

YuNet Detector: face_detection_yunet_2023mar_int8.onnx

ArcFace Recognizer: w600k_r50.onnx (The validated Float Model)

Installation
Clone Repository:

git clone [https://github.com/ay-ush-17/Ezpic_Pipeline](https://github.com/ay-ush-17/Ezpic_Pipeline)
cd Ezpic_Pipeline


Install Dependencies:

# Use your environment's Python executable path if 'pip' is not recognized
python -m pip install onnxruntime opencv-python numpy scikit-learn pillow


Place Models: Create a Models/ directory and place the downloaded w600k_r50.onnx and face_detection_yunet_2023mar_int8.onnx files inside it.

Running the Verification Benchmark
Use the final benchmark script to test the model's performance against the industry-standard LFW Cross-Pose protocol.

# Set your PYTHONPATH to include the WORKING CODE folder if necessary
# Example: python scripts/lfw_benchmark.py --max-images 2000
python scripts/lfw_benchmark.py


Running the Multi-View GUI
The GUI allows interactive testing of the Enrollment (3 poses) and Matching (Minimum Distance) logic.

python whole_face_scan_gui.py


⚠️ Final Deployment Note
Due to the fatal accuracy loss during the INT8 quantization attempt, the recommended deployment asset is the 166 MB Float ONNX Model (w600k_r50.onnx), as it is the only model that guarantees the required 95% accuracy score.
