import cv2
import numpy as np
from .detector import YuNetDetector
from .embedder import ArcFaceEmbedder
from .vector_db import VectorDB


def simple_liveness_check(face_img: np.ndarray) -> bool:
    """A lightweight liveness heuristic: checks for sufficient variance and eyes opening via brightness."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    if gray.std() < 10:
        return False
    # crude eye brightness check: look for darker regions near top half
    h = gray.shape[0]
    top = gray[: h // 2]
    if top.mean() < 30:
        return False
    return True


class FacePipeline:
    def __init__(self, yunet_path: str, arcface_path: str, db_path=None):
        self.detector = YuNetDetector(yunet_path)
        self.embedder = ArcFaceEmbedder(arcface_path)
        self.db = VectorDB(dim=512, index_path=db_path or 'face_index.ann', map_path='face_map.json')

    def detect_and_embed(self, image: np.ndarray, conf=0.6):
        dets = self.detector.detect(image, conf_threshold=conf)
        results = []
        for d in dets:
            x1, y1, x2, y2 = d['box']
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            live = simple_liveness_check(crop)
            emb = None
            if live:
                emb = self.embedder.embed(crop)
            results.append({'box': d['box'], 'score': d['score'], 'liveness': live, 'embedding': emb})
        return results

    def enroll(self, subject_id: str, images: list, metadata=None, consent=True):
        # images: list of BGR face crops
        eids = []
        for img in images:
            emb = self.embedder.embed(img)
            eid = self.db.add(emb, subject_id, metadata=metadata, consent=consent)
            eids.append(eid)
        # after adding multiple views, build index
        self.db.build()
        return eids

    def query(self, image: np.ndarray, top_k=5):
        res = self.detect_and_embed(image)
        matches = []
        for r in res:
            if r['embedding'] is None:
                matches.append({'box': r['box'], 'match': None})
                continue
            ids, dists = self.db.search(r['embedding'], top_k=top_k)
            # map ids back to subject
            subject_hits = []
            for sid, info in self.db.map.items():
                for eid in info.get('eids', []):
                    if eid in ids:
                        subject_hits.append({'subject_id': sid, 'eid': eid, 'metadata': info.get('metadata')})
            matches.append({'box': r['box'], 'matches': subject_hits, 'dists': dists})
        return matches
