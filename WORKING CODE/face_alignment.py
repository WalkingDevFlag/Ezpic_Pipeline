"""
Face alignment helpers copied for WORKING CODE subset.
- Provides `align_and_crop(image, landmarks, output_size=112)` returning an aligned face crop (numpy array).

Notes:
- This is a trimmed, dependency-free version of the alignment utilities from the main repo.
- Landmark order assumed: [x0,y0,x1,y1,...] where landmarks[0:4] contain left-eye and right-eye coords.
"""

import numpy as np
import cv2

def _eye_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def align_and_crop(image, landmarks, output_size=112):
    """Aligns the face given 5-point landmarks (x0,y0,...). Returns a warped crop.

    Args:
        image: input BGR image (numpy array)
        landmarks: flat list/array of 10 floats [x0,y0,...,x4,y4]
        output_size: desired output size (square)
    Returns:
        aligned_face: BGR uint8 image of shape (output_size, output_size, 3)
    """
    # We expect left-eye at (x0,y0) and right-eye at (x1,y1)
    # For robustness accept both float and int landmarks
    lm = np.asarray(landmarks).reshape(-1, 2).astype(np.float32)
    left_eye = lm[0]
    right_eye = lm[1]

    # Compute angle and scale
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    dist = _eye_distance(left_eye, right_eye)

    # Desired positions: eyes horizontally centered
    desired_left = np.array([output_size * 0.35, output_size * 0.4], dtype=np.float32)
    desired_right = np.array([output_size * 0.65, output_size * 0.4], dtype=np.float32)
    desired_dist = np.linalg.norm(desired_right - desired_left)
    scale = desired_dist / (dist + 1e-6)

    # Rotation + scale matrix around left eye
    M = cv2.getRotationMatrix2D(tuple(left_eye), angle, scale)

    # After rotation+scale, transform left_eye to its desired position
    transformed_left = M.dot(np.array([left_eye[0], left_eye[1], 1.0]))
    tx = desired_left[0] - transformed_left[0]
    ty = desired_left[1] - transformed_left[1]
    M[:, 2] += (tx, ty)

    aligned = cv2.warpAffine(image, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned


def align_face_simple(image, landmarks, output_size=112):
    """Compatibility shim used by WORKING CODE modules.

    Calls align_and_crop (which returns a BGR numpy array) and converts result to RGB
    which some modules expect.

    Args:
        image: BGR numpy image
        landmarks: flat list/array of 10 floats (x0,y0,...)
        output_size: desired square output size

    Returns:
        aligned_rgb: RGB numpy array (output_size, output_size, 3), dtype=uint8
    """
    aligned_bgr = align_and_crop(image, landmarks, output_size=output_size)
    try:
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        # If conversion fails, return the BGR array to avoid breaking callers
        return aligned_bgr
    return aligned_rgb
