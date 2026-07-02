"""Score-vector validation helpers."""

from __future__ import annotations

import numpy as np


def validate_score_matrix(scores: np.ndarray, *, target_count: int) -> None:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D matrix")
    if scores.shape[1] != target_count:
        raise ValueError(
            f"score matrix has {scores.shape[1]} columns; expected {target_count}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite (no NaN or Inf)")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("scores must lie within [0, 1]")
