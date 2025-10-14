import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Dict


class YuNetDetector:
    """Robust YuNet detector wrapper.

    Prefers OpenCV's FaceDetectorYN wrapper when available (it handles
    the model I/O and landmark/bbox formatting). If the OpenCV wrapper
    is not available, falls back to an ONNXRuntime-based decoder that
    understands the multi-head YuNet outputs (obj_*, bbox_*).

    API: detect(image, conf_threshold=0.6, top_k=1) -> List[{'box':(x1,y1,x2,y2),'score':float}]
    """

    def __init__(self, model_path: str, providers=None, preferred_input_size=(640, 640)):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.model_path = str(model_path)
        self.preferred_input_size = preferred_input_size

        # Try to initialize OpenCV FaceDetectorYN if available
        self.cv_wrapper = None
        try:
            if hasattr(cv2, 'FaceDetectorYN'):
                try:
                    # Try named-arg creation first (newer OpenCV)
                    self.cv_wrapper = cv2.FaceDetectorYN.create(
                        self.model_path,
                        '',
                        self.preferred_input_size,
                        score_threshold=0.35,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                except Exception:
                    # Fallback to positional args for older builds
                    self.cv_wrapper = cv2.FaceDetectorYN.create(
                        self.model_path, '', self.preferred_input_size, 0.35, 0.3, 5000
                    )
        except Exception:
            self.cv_wrapper = None

        # Always prepare an ONNXRuntime session fallback
        try:
            self.sess = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.sess.get_inputs()[0].name
            in_shape = self.sess.get_inputs()[0].shape
            try:
                _, c, h, w = in_shape
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    self.input_size = (w, h)
                else:
                    self.input_size = None
            except Exception:
                self.input_size = None
        except Exception:
            # If ONNX session creation fails, set to None and rely on cv wrapper only
            self.sess = None
            self.input_name = None
            self.input_size = None

    def detect(self, image: np.ndarray, conf_threshold=0.6, top_k=1) -> List[Dict]:
        """Detect faces and return standardized detection dicts.

        image: BGR uint8 numpy array
        """
        orig_h, orig_w = image.shape[:2]

        # 1) Prefer OpenCV wrapper if available
        if self.cv_wrapper is not None:
            try:
                # FaceDetectorYN expects width,height ordering for setInputSize
                try:
                    self.cv_wrapper.setInputSize((orig_w, orig_h))
                except Exception:
                    # some builds expose setInputSize with tuple reversed or not available
                    pass
                ret, faces = self.cv_wrapper.detect(image)
                detections = []
                if faces is None or faces.size == 0:
                    return detections
                for i in range(min(len(faces), top_k)):
                    f = faces[i]
                    # OpenCV FaceDetectorYN returns [x, y, w, h, score, ...landmarks...]
                    try:
                        x, y, w_box, h_box = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        score = float(f[4]) if f.shape[0] > 4 else 1.0
                    except Exception:
                        # Unexpected format, skip
                        continue
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(orig_w - 1, int(x + w_box))
                    y2 = min(orig_h - 1, int(y + h_box))
                    detections.append({'box': (x1, y1, x2, y2), 'score': score})
                return detections
            except Exception:
                # If cv wrapper errors, fall back to ONNX path
                pass

        # 2) ONNXRuntime fallback path: decode common YuNet multi-head outputs
        if self.sess is None:
            return []

        # prepare input
        inp_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        if getattr(self, 'input_size', None) is not None:
            inp_resized = cv2.resize(inp_img, self.input_size)
        else:
            # resize to preferred size for consistent decoding
            inp_resized = cv2.resize(inp_img, self.preferred_input_size)
        blob = inp_resized.transpose(2, 0, 1)[None, ...].astype('float32')

        outs = self.sess.run(None, {self.input_name: blob})
        names = [o.name for o in self.sess.get_outputs()]
        out_map = {n: v for n, v in zip(names, outs)}

        # prefer coarse heads (_32, _16, _8)
        obj = None
        bbox = None
        for suf in ['_32', '_16', '_8']:
            if f'obj{suf}' in out_map and f'bbox{suf}' in out_map:
                obj = out_map[f'obj{suf}'][0].reshape(-1)
                bbox = out_map[f'bbox{suf}'][0].reshape(-1, 4)
                break

        detections = []
        if obj is None:
            # try to find any 2D or 3D array that looks like detections
            for out in outs:
                if isinstance(out, np.ndarray):
                    if out.ndim == 3 and out.shape[2] >= 6:
                        dets = out[0]
                        for d in dets:
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
                                scale_x = orig_w / float(inp_resized.shape[1])
                                scale_y = orig_h / float(inp_resized.shape[0])
                                x1 = max(0, int(x0 * scale_x))
                                y1 = max(0, int(y0 * scale_y))
                                x2 = min(orig_w - 1, int(x1r * scale_x))
                                y2 = min(orig_h - 1, int(y1r * scale_y))
                            detections.append({'box': (x1, y1, x2, y2), 'score': score})
                        return detections
        else:
            idxs = obj.argsort()[::-1]
            for idx in idxs[:top_k]:
                score = float(obj[idx])
                if score < conf_threshold:
                    continue
                x_c, y_c, bw, bh = bbox[idx]
                if max(abs(x_c), abs(y_c), abs(bw), abs(bh)) <= 1.01:
                    cx = x_c * orig_w
                    cy = y_c * orig_h
                    bw_px = bw * orig_w
                    bh_px = bh * orig_h
                    x1 = int(max(0, cx - bw_px / 2))
                    y1 = int(max(0, cy - bh_px / 2))
                    x2 = int(min(orig_w - 1, cx + bw_px / 2))
                    y2 = int(min(orig_h - 1, cy + bh_px / 2))
                else:
                    # already in pixels relative to resized
                    x1 = int(max(0, x_c))
                    y1 = int(max(0, y_c))
                    x2 = int(min(orig_w - 1, x_c + bw))
                    y2 = int(min(orig_h - 1, y_c + bh))
                detections.append({'box': (x1, y1, x2, y2), 'score': score})
            return detections

        return detections
