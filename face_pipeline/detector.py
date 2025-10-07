import onnxruntime as ort
import numpy as np
import cv2


class YuNetDetector:
    """Simple wrapper around YuNet ONNX model.

    Assumes model takes BGR image input and outputs detections in the format
    [num, 6] -> [x1, y1, x2, y2, score, label] or similar. We adapt to the
    common Yunet output used in OpenCV's implementation.
    """

    def __init__(self, model_path: str, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        # Try to infer the static input size expected by the model (C,H,W)
        in_shape = self.sess.get_inputs()[0].shape
        # Typical Yunet input shape: [1, 3, 640, 640]
        try:
            # shape may contain 'None' or symbolic dims; ensure ints exist
            _, c, h, w = in_shape
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                self.input_size = (w, h)
            else:
                self.input_size = None
        except Exception:
            self.input_size = None

    def detect(self, image: np.ndarray, conf_threshold=0.6):
        """Detect faces. Returns list of dicts: {box: (x1,y1,x2,y2), score}

        image: BGR uint8 numpy array
        """
        orig_h, orig_w = image.shape[:2]
        img = image.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model input size if it's fixed
        resized_w, resized_h = orig_w, orig_h
        if getattr(self, 'input_size', None) is not None:
            resized_w, resized_h = self.input_size
            img_resized = cv2.resize(img, (resized_w, resized_h))
        else:
            # If model accepts dynamic size, use original
            img_resized = img

        inp = (img_resized / 255.0).transpose(2, 0, 1)[None, ...].astype(np.float32)

        outs = self.sess.run(None, {self.input_name: inp})
        # Typical Yunet outputs: loc, conf, lmk or single detections array
        # Try to interpret common formats
        detections = []
        # If single output with detections
        for out in outs:
            if isinstance(out, np.ndarray) and out.ndim == 3 and out.shape[2] >= 6:
                dets = out[0]
                for d in dets:
                    score = float(d[4])
                    if score < conf_threshold:
                        continue
                    # d[0..3] may be normalized [0,1] or pixels relative to resized image
                    x0, y0, x1r, y1r = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                    # decide scaling: if values <= 1.0 treat as normalized
                    if max(x0, y0, x1r, y1r) <= 1.01:
                        x1 = max(0, int(x0 * orig_w))
                        y1 = max(0, int(y0 * orig_h))
                        x2 = min(orig_w - 1, int(x1r * orig_w))
                        y2 = min(orig_h - 1, int(y1r * orig_h))
                    else:
                        # values are in pixels relative to resized image
                        scale_x = orig_w / float(resized_w)
                        scale_y = orig_h / float(resized_h)
                        x1 = max(0, int(x0 * scale_x))
                        y1 = max(0, int(y0 * scale_y))
                        x2 = min(orig_w - 1, int(x1r * scale_x))
                        y2 = min(orig_h - 1, int(y1r * scale_y))
                    detections.append({'box': (x1, y1, x2, y2), 'score': score})
                return detections

        # Fallback: look for (N,6) output
        for out in outs:
            if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 6:
                for d in out:
                    score = float(d[4])
                    if score < conf_threshold:
                        continue
                    x0, y0, x1r, y1r = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                    if max(x0, y0, x1r, y1r) <= 1.01:
                        x1 = max(0, int(x0 * orig_w))
                        y1 = max(0, int(y0 * orig_h))
                        x2 = min(orig_w - 1, int(x1r * orig_w))
                        y2 = min(orig_h - 1, int(y1r * orig_h))
                    else:
                        scale_x = orig_w / float(resized_w)
                        scale_y = orig_h / float(resized_h)
                        x1 = max(0, int(x0 * scale_x))
                        y1 = max(0, int(y0 * scale_y))
                        x2 = min(orig_w - 1, int(x1r * scale_x))
                        y2 = min(orig_h - 1, int(y1r * scale_y))
                    detections.append({'box': (x1, y1, x2, y2), 'score': score})
                return detections

        return detections
