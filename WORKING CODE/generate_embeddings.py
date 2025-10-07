"""
Generate embeddings helper for WORKING CODE subset.
Provides a thin wrapper around the ArcFace ONNX model used by the multi-face search script.

Exports:
- load_embedder(): returns an object with `session` and `run(image)` semantics
- calculate_similarity(a, b): returns cosine similarity
- calculate_distance(a, b): returns Euclidean distance

This file intentionally keeps dependencies minimal (onnxruntime, numpy, cv2).
"""

import os
import numpy as np
import onnxruntime as ort
import cv2

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "w600k_r50.onnx")


def _preprocess_arcface(bgr_image):
    # Expect BGR image 112x112 -> RGB float32 0-1 normalized with mean/std (if any)
    img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


class ArcFaceEmbedder:
    def __init__(self, model_path=None, provider=None):
        if model_path is None:
            model_path = MODEL_PATH
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        providers = [provider] if provider else None
        self.session = ort.InferenceSession(model_path, sess_opts, providers=providers)
        # Get input name
        self.input_name = self.session.get_inputs()[0].name

    def run(self, bgr_image):
        x = _preprocess_arcface(bgr_image)
        out = self.session.run(None, {self.input_name: x})[0]
        vec = np.array(out).reshape(-1)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec


def load_embedder():
    return ArcFaceEmbedder()


def calculate_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def calculate_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b))
