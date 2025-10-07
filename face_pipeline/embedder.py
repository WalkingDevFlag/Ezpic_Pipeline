import onnxruntime as ort
import numpy as np
import cv2


class ArcFaceEmbedder:
    def __init__(self, model_path: str, input_size=(112, 112), providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        # face_img: BGR uint8
        img = cv2.resize(face_img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        inp = np.transpose(img, (2, 0, 1)).astype(np.float32)
        inp = np.expand_dims(inp, axis=0)
        return inp

    def embed(self, face_img: np.ndarray) -> np.ndarray:
        inp = self.preprocess(face_img)
        outs = self.sess.run(None, {self.input_name: inp})
        # Usually output is (1,512)
        emb = None
        for out in outs:
            if isinstance(out, np.ndarray) and out.ndim == 2:
                emb = out[0]
                break
        if emb is None:
            # pick first ndarray and flatten
            for out in outs:
                if isinstance(out, np.ndarray):
                    emb = out.flatten()[:512]
                    break
        emb = emb.astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
