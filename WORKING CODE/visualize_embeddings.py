"""
Minimal visualization helpers used by the small WORKING CODE package.
These functions are optional; they provide simple plotting utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_complete_pipeline(image, aligned, embedding):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    try:
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception:
        axs[0].imshow(image)
    axs[0].set_title("Original")
    try:
        axs[1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    except Exception:
        axs[1].imshow(aligned)
    axs[1].set_title("Aligned")
    axs[2].plot(embedding)
    axs[2].set_title("Embedding")
    plt.tight_layout()
    plt.show()


def visualize_embedding_vector(vec):
    plt.figure(figsize=(6, 2))
    plt.plot(vec)
    plt.title("ArcFace Embedding")
    plt.show()


def compare_embeddings(a, b):
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    dist = np.linalg.norm(a - b)
    print(f"Similarity: {sim:.4f}, Distance: {dist:.4f}")
