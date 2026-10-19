# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Layer 2: k nearest neighbours over normalized embeddings of labelled prompts."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_MIN_WEIGHT = 1e-6


class KnnIndex:
    """Vectors (unit length) with a COMPLEX label (0.0 or 1.0 or a score in between) for each."""

    def __init__(self, vectors: NDArray[np.float32], labels: NDArray[np.float32]) -> None:
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("vectors must be a non-empty 2-D array")
        if labels.shape != (len(vectors),):
            raise ValueError("labels must have one value per vector")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        self._labels = labels.astype(np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "KnnIndex":
        """Load an index saved with np.savez (arrays 'vectors' and 'labels')."""
        with np.load(path) as data:
            return cls(data["vectors"], data["labels"])

    @property
    def size(self) -> int:
        """Number of indexed prompts."""
        return len(self._vectors)

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return int(self._vectors.shape[1])

    def score(self, query: Sequence[float], k: int) -> float:
        """Similarity-weighted mean label of the k nearest neighbours, in [0, 1]."""
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            msg = f"query has dimension {q.shape}, index has {self.dim}"
            raise ValueError(msg)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            raise ValueError("query vector is zero")
        sims = self._vectors @ (q / norm)
        k = min(k, self.size)
        idx = np.argpartition(-sims, k - 1)[:k]
        weights = np.clip(sims[idx], _MIN_WEIGHT, None)
        return float((weights * self._labels[idx]).sum() / weights.sum())
