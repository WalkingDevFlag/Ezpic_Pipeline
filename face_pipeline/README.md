# Face Retrieval Pipeline (YuNet + ArcFace)

This small project demonstrates loading YuNet (face detection) and ArcFace (face embedding) ONNX models, testing them individually, and integrating them into a simple pipeline with a vector DB (Annoy) and lightweight liveness checking.

Key points:
- Detector: `face_detection_yunet_2023mar_int8.onnx` (place in `Models/`)
- Embedder: `w600k_r50.onnx` (place in `Models/`)
- Vector DB: Annoy-based index with simple metadata store
- GDPR/CCPA: consent flag stored per subject; erase operation is a stub that requires external reindexing of embeddings (Annoy doesn't support deletion)

Quick start
1. Create a venv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r face_pipeline\requirements.txt
```

2. Put a sample image at `face_pipeline/sample_face.jpg` and ensure models are in `Models/` relative to workspace root.

3. Run the example:

```powershell
python -m face_pipeline.example

Desktop GUI (Tkinter)
1. Create and activate the virtual environment and install requirements:

```powershell
# from D:\MY WORK\pipeline
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r .\face_pipeline\requirements.txt
```

2. Run the desktop GUI:

```powershell
python -m face_pipeline.multi_view_gui
```

This opens a desktop window that guides you through multi-view enrollment (frontal, left, right), provides visual feedback for each capture, and allows you to test enrollments with arbitrary images.
```

Next steps and notes
- Improve liveness with proper camera-based cues or a trained liveness model.
- Store embeddings persistently to allow true GDPR deletion (rebuild index after removing embeddings).
- For sub-100ms on-device performance, prefer running with a hardware-accelerated provider (ONNX Runtime with OpenVINO, CUDA, or NNAPI) and quantized models.
