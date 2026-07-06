"""Apply a frozen global classification threshold to model scores."""

from __future__ import annotations

import numpy as np

from repotriage.inference.models import LabelScore, PredictedLabel
from repotriage.threshold_policy.sweep import predictions_from_scores


def apply_classification_threshold(
    *,
    labels: list[str],
    y_score: np.ndarray,
    threshold: float,
    threshold_basis_points: int,
) -> tuple[list[LabelScore], list[PredictedLabel], np.ndarray]:
    """Apply the frozen threshold and build ordered score and prediction lists."""
    y_pred = predictions_from_scores(y_score.reshape(1, -1), threshold)[0]
    scores = [
        LabelScore(label=label, score=float(y_score[index]))
        for index, label in enumerate(labels)
    ]
    predicted: list[PredictedLabel] = []
    for index, label in enumerate(labels):
        if y_pred[index]:
            predicted.append(
                PredictedLabel(label=label, score=float(y_score[index]))
            )
    predicted.sort(key=lambda item: (-item.score, labels.index(item.label)))
    return scores, predicted, y_pred
