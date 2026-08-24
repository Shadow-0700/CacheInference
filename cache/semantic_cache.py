try:
    import hnswlib  # type: ignore
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

import numpy as np
from fastembed import TextEmbedding  # type: ignore
from typing import Optional, Tuple, Dict, Any

class SemanticCache:
    def __init__(self, dim: int = 384, max_elements: int = 10000, model_name: str = "BAAI/bge-small-en-v1.5", threads: Optional[int] = 8):
        self.dim = dim
        self.max_elements = max_elements
        self.model = TextEmbedding(model_name=model_name, threads=threads)
        
        self.has_hnsw = HAS_HNSWLIB
        if self.has_hnsw:
            self.index = hnswlib.Index(space="cosine", dim=self.dim)
            self.index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
            self.index.set_ef(50)
        else:
            self.vectors: Dict[int, np.ndarray] = {}

        self.entries: Dict[int, Dict[str, Any]] = {}
        self.current_count = 0

    def embed(self, prompt: str) -> np.ndarray:
        embeddings = list(self.model.embed([prompt]))
        emb = np.array(embeddings[0], dtype=np.float32)
        return emb

    def search(self, prompt: str, embedding: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[Any], float, Optional[np.ndarray]]:
        if self.current_count == 0:
            if embedding is None:
                embedding = self.embed(prompt)
            return None, None, 0.0, embedding

        if embedding is None:
            embedding = self.embed(prompt)

        if self.has_hnsw:
            labels, distances = self.index.knn_query(embedding, k=1)
            label = int(labels[0][0])
            distance = float(distances[0][0])
            similarity = 1.0 - distance
        else:
            # Cosine similarity fallback
            best_label = None
            best_sim = -1.0
            norm_e = np.linalg.norm(embedding)
            for lbl, vec in self.vectors.items():
                denom = norm_e * np.linalg.norm(vec)
                sim = float(np.dot(embedding, vec) / denom) if denom > 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_label = lbl
            label = best_label if best_label is not None else 0
            similarity = best_sim

        entry = self.entries.get(label)
        if entry:
            return entry["prompt"], entry["response"], similarity, embedding
        return None, None, similarity, embedding

    def add(self, prompt: str, embedding: np.ndarray, response: Any) -> None:
        label = self.current_count

        if self.has_hnsw:
            if self.current_count >= self.max_elements:
                self.max_elements *= 2
                self.index.resize_index(self.max_elements)
            self.index.add_items(np.array([embedding], dtype=np.float32), np.array([label]))
        else:
            self.vectors[label] = embedding

        self.entries[label] = {
            "prompt": prompt,
            "response": response,
            "embedding": embedding
        }
        self.current_count += 1
