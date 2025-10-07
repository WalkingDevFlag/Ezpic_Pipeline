import os
import cv2
from .detector import YuNetDetector
from .embedder import ArcFaceEmbedder

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'Models')


def test_detector(yunet_path, img_path):
    det = YuNetDetector(yunet_path)
    img = cv2.imread(img_path)
    dets = det.detect(img, conf_threshold=0.5)
    print('Detections:', dets)


def test_embedder(arcface_path, img_path):
    embd = ArcFaceEmbedder(arcface_path)
    img = cv2.imread(img_path)
    # assume full image is a face for quick test
    emb = embd.embed(img)
    print('Embedding shape:', emb.shape)
    print('First 8 dims:', emb[:8])


if __name__ == '__main__':
    yunet = os.path.join(os.path.dirname(__file__), '..', 'Models', 'face_detection_yunet_2023mar_int8.onnx')
    arcface = os.path.join(os.path.dirname(__file__), '..', 'Models', 'w600k_r50.onnx')
    sample = os.path.join(os.path.dirname(__file__), '..', 'sample_face.jpg')
    if not os.path.exists(sample):
        print('Place a sample face image at', sample)
    else:
        print('Testing detector...')
        test_detector(yunet, sample)
        print('Testing embedder...')
        test_embedder(arcface, sample)
