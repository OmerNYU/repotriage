"""TF-IDF feature extraction for retrieval indexing (train-only fit)."""

from __future__ import annotations

from repotriage.baseline.features import build_vectorizer, fit_vectorizer, transform_texts
from repotriage.baseline.models import TfidfParams

__all__ = ["TfidfParams", "build_vectorizer", "fit_vectorizer", "transform_texts"]
