"""Tests for issue-level confidence computation."""

from __future__ import annotations

import numpy as np
import pytest

from repotriage.abstention_policy.confidence import (
    build_issue_confidence_table,
    handled_mask_for_abstention_threshold,
    issue_confidence_from_scores,
)
from repotriage.baseline.scores import validate_score_matrix


def test_no_predicted_labels_forces_none_confidence() -> None:
    y_score = np.array([0.1, 0.2, 0.15], dtype=np.float64)
    y_pred = np.array([0, 0, 0], dtype=np.int8)
    assert issue_confidence_from_scores(y_score, y_pred) is None


def test_issue_confidence_is_max_among_predicted_labels() -> None:
    y_score = np.array([0.95, 0.4, 0.7], dtype=np.float64)
    y_pred = np.array([1, 1, 0], dtype=np.int8)
    assert issue_confidence_from_scores(y_score, y_pred) == pytest.approx(0.95)


def test_scores_below_classification_threshold_ignored_for_confidence() -> None:
    table = build_issue_confidence_table(
        issue_ids=[1],
        y_score=np.array([[0.95, 0.42, 0.35]], dtype=np.float64),
        classification_threshold=0.5,
    )
    assert table.confidences == [pytest.approx(0.95)]
    assert table.forced_abstention_mask.tolist() == [False]


def test_forced_abstention_when_no_labels_predicted() -> None:
    table = build_issue_confidence_table(
        issue_ids=[1, 2],
        y_score=np.array([[0.1, 0.2], [0.8, 0.2]], dtype=np.float64),
        classification_threshold=0.5,
    )
    assert table.confidences[0] is None
    assert table.forced_abstention_mask[0]
    assert table.confidences[1] == pytest.approx(0.8)


def test_handled_mask_is_inclusive_at_abstention_threshold() -> None:
    table = build_issue_confidence_table(
        issue_ids=[1],
        y_score=np.array([[0.6, 0.2]], dtype=np.float64),
        classification_threshold=0.5,
    )
    assert handled_mask_for_abstention_threshold(table, 0.6).tolist() == [True]
    assert handled_mask_for_abstention_threshold(table, 0.61).tolist() == [False]


def test_nan_scores_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_score_matrix(np.array([[0.5, np.nan]], dtype=np.float64), target_count=2)


def test_vector_dimension_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="columns"):
        validate_score_matrix(np.array([[0.5]], dtype=np.float64), target_count=2)
