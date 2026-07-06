"""Score one issue using a loaded baseline classifier artifact."""

from __future__ import annotations

import numpy as np

from repotriage.baseline.models_ml import TfidfMultiLabelLogRegModel
from repotriage.baseline.scores import validate_score_matrix


def score_issue(model: TfidfMultiLabelLogRegModel, feature_text: str) -> np.ndarray:
    """Return per-label probability scores for one feature-text string."""
    scores = model.predict_proba_matrix([feature_text])
    validate_score_matrix(scores, target_count=len(model.labels))
    return scores[0]
