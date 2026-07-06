"""Tests for inference threshold application."""

from __future__ import annotations

import numpy as np

from repotriage.inference.thresholding import apply_classification_threshold


def test_predicted_labels_sorted_by_score_then_label_order() -> None:
    labels = ["Bug", "Indexing", "Docs"]
    y_score = np.array([0.5, 0.8, 0.7], dtype=np.float64)
    _scores, predicted, y_pred = apply_classification_threshold(
        labels=labels,
        y_score=y_score,
        threshold=0.39,
        threshold_basis_points=39,
    )
    assert y_pred.tolist() == [1, 1, 1]
    assert [item.label for item in predicted] == ["Indexing", "Docs", "Bug"]


def test_no_labels_below_threshold() -> None:
    labels = ["Bug", "Docs"]
    y_score = np.array([0.1, 0.2], dtype=np.float64)
    scores, predicted, y_pred = apply_classification_threshold(
        labels=labels,
        y_score=y_score,
        threshold=0.39,
        threshold_basis_points=39,
    )
    assert len(scores) == 2
    assert predicted == []
    assert y_pred.tolist() == [0, 0]
