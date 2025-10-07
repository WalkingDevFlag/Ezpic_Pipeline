"""
Complete Face Recognition Pipeline Orchestrator
================================================

This script orchestrates the entire pipeline:
1. YuNet Face Detection
2. Face Alignment (112×112)
3. MobileFaceNet Embedding Generation
4. Visualization

Usage:
    from complete_pipeline import process_single_face, compare_two_faces
    
    # Process one face
    embedding = process_single_face("image.jpg")
    
    # Compare two faces
    similarity = compare_two_faces("face1.jpg", "face2.jpg")

Author: EZ pic Face Recognition Pipeline
Date: October 5, 2025
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

# Import our modular components
from face_alignment import align_and_crop
from generate_embeddings import load_embedder, calculate_similarity, calculate_distance
from visualize_embeddings import (
    visualize_complete_pipeline,
    visualize_embedding_vector,
    compare_embeddings
)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar_int8.onnx")
MOBILEFACENET_PATH = os.path.join(BASE_DIR, "models", "MobileFaceNet_9925_9680.pb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GLOBAL MODELS (Load once, use many times) ---
_yunet_detector = None
_embedder = None

def load_yunet():
    """Load YuNet detector (singleton pattern)."""
    global _yunet_detector
    if _yunet_detector is None:
        _yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )
        print("✅ YuNet detector loaded")
    return _yunet_detector


def load_mobilefacenet():
    """Load MobileFaceNet embedder (singleton pattern)."""
    global _embedder
    if _embedder is None:
        _embedder = load_embedder()
    return _embedder
