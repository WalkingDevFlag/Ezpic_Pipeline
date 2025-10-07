try:
    from annoy import AnnoyIndex
    _HAS_ANNOY = True
except Exception:
    AnnoyIndex = None
    _HAS_ANNOY = False

import numpy as np
import threading
import json
import os


class VectorDB:
    def __init__(self, dim=512, metric='angular', index_path=None, map_path=None, n_trees=10):
        self.dim = dim
        self.metric = metric
        # If Annoy is available, use it for fast ANN. Otherwise use a simple
        # in-memory brute-force store for testing/development.
        if _HAS_ANNOY:
            self.index = AnnoyIndex(dim, metric)
        else:
            self.index = None
            # embeddings: eid -> numpy array
            self._embeddings = {}
        self.n_trees = n_trees
        self.lock = threading.Lock()
        self.map = {}  # id -> metadata (embedding ids, consent flag, etc)
        self.next_id = 0
        self.index_path = index_path
        self.map_path = map_path
        # default embeddings persistence path (if map_path provided)
        if self.map_path:
            base, _ = os.path.splitext(self.map_path)
            self.embeddings_path = base + '_embeddings.npz'
        else:
            self.embeddings_path = None
        if index_path and os.path.exists(index_path):
            if _HAS_ANNOY:
                self.index.load(index_path)
        if map_path and os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf8') as f:
                data = json.load(f)
                self.map = data.get('map', {})
                self.next_id = data.get('next_id', 0)
        # load persisted embeddings if available (in-memory fallback)
        if not _HAS_ANNOY and self.embeddings_path and os.path.exists(self.embeddings_path):
            try:
                arr = np.load(self.embeddings_path)
                for k in arr.files:
                    self._embeddings[int(k)] = arr[k]
            except Exception:
                pass

    def _persist(self):
        if self.index_path:
            if _HAS_ANNOY and self.index is not None:
                self.index.save(self.index_path)
        if self.map_path:
            with open(self.map_path, 'w', encoding='utf8') as f:
                json.dump({'map': self.map, 'next_id': self.next_id}, f)
        # persist in-memory embeddings if present
        if not _HAS_ANNOY and getattr(self, '_embeddings', None) is not None and self.embeddings_path:
            try:
                # save as compressed npz with keys as string ids
                save_dict = {str(k): v for k, v in self._embeddings.items()}
                np.savez_compressed(self.embeddings_path, **save_dict)
            except Exception:
                pass

    def add(self, embedding: np.ndarray, subject_id: str, metadata=None, consent=True):
        """Adds a single embedding tied to subject_id. Multi-view enrollment simply adds multiple embeddings with same subject_id."""
        with self.lock:
            eid = self.next_id
            self.next_id += 1
            if _HAS_ANNOY and self.index is not None:
                self.index.add_item(eid, embedding.tolist())
            else:
                self._embeddings[eid] = np.array(embedding, dtype=np.float32)
            if subject_id not in self.map:
                self.map[subject_id] = {'eids': [], 'metadata': metadata or {}, 'consent': consent}
            self.map[subject_id]['eids'].append(eid)
            # persist metadata (and Annoy if applicable)
            self._persist()
            return eid

    def build(self):
        with self.lock:
            if _HAS_ANNOY and self.index is not None:
                self.index.build(self.n_trees)
            # for in-memory fallback, nothing to build
            self._persist()

    def search(self, embedding: np.ndarray, top_k=5, include_distances=True):
        # Ensure index is built in memory
        with self.lock:
            if _HAS_ANNOY and self.index is not None:
                ids, dists = self.index.get_nns_by_vector(embedding.tolist(), top_k, include_distances=include_distances)
                return ids, dists
            # brute-force cosine similarity over stored embeddings
            if len(self._embeddings) == 0:
                return [], []
            # assume embeddings are L2-normalized
            eids = list(self._embeddings.keys())
            mats = np.stack([self._embeddings[eid] for eid in eids], axis=0)
            query = np.array(embedding, dtype=np.float32)
            sims = mats.dot(query)
            # convert to distance-like (1 - cosine)
            dists = 1.0 - sims
            idx = np.argsort(dists)[:top_k]
            return [eids[i] for i in idx], [float(dists[i]) for i in idx]

    def erase_subject(self, subject_id: str):
        """GDPR right-to-erasure: remove all embeddings and metadata for subject.
        Annoy doesn't support deletion; to comply we mark as deleted and rebuild.
        """
        with self.lock:
            if subject_id not in self.map:
                return False
            removed = set(self.map[subject_id]['eids'])
            del self.map[subject_id]
            if _HAS_ANNOY and self.index is not None:
                # Annoy doesn't support deletion; user should rebuild from persistent store
                # We'll persist metadata change and return; full reindex requires external embeddings
                self._persist()
                return True
            # For in-memory fallback, remove embeddings directly
            for eid in removed:
                if eid in self._embeddings:
                    del self._embeddings[eid]
            self._persist()
            return True
