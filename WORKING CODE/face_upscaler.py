"""
Face Upscaler for Low-Resolution Faces (copied into WORKING CODE)
"""

import cv2
import numpy as np


def upscale_small_face(image, bbox, landmarks, min_size=150, target_size=224):
    x, y, w, h = bbox
    face_size = min(w, h)
    if face_size >= min_size:
        return image, bbox, landmarks

    scale_factor = target_size / face_size
    padding_ratio = 0.3
    padding_w = int(w * padding_ratio)
    padding_h = int(h * padding_ratio)

    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(image.shape[1], x + w + padding_w)
    y2 = min(image.shape[0], y + h + padding_h)

    face_region = image[y1:y2, x1:x2].copy()
    if face_region.size == 0:
        return image, bbox, landmarks

    upscaled_face = upscale_face_region(face_region, scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    enhanced_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    new_bbox = np.array([
        int(x * scale_factor),
        int(y * scale_factor),
        int(w * scale_factor),
        int(h * scale_factor)
    ], dtype=int)

    new_landmarks = landmarks.copy().astype(np.float32)
    new_landmarks[:, 0] *= scale_factor
    new_landmarks[:, 1] *= scale_factor

    return enhanced_image, new_bbox, new_landmarks


def upscale_face_region(face_region, scale_factor):
    new_width = int(face_region.shape[1] * scale_factor)
    new_height = int(face_region.shape[0] * scale_factor)
    upscaled = cv2.resize(face_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    upscaled = sharpen_image(upscaled)
    upscaled = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
    return upscaled


def sharpen_image(image, amount=1.0):
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    result = cv2.addWeighted(image, 1.0 - amount, sharpened, amount, 0)
    return result


def preprocess_small_face(image, bbox, landmarks, min_size=100, target_size=200):
    x, y, w, h = bbox
    face_size = min(w, h)
    if face_size >= min_size:
        return image, bbox, landmarks
    enhanced_image, new_bbox, new_landmarks = upscale_small_face(
        image, bbox, landmarks, min_size, target_size
    )
    return enhanced_image, new_bbox, new_landmarks


if __name__ == "__main__":
    print("Face Upscaler Module (WORKING CODE)")
