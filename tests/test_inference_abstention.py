"""Tests for single-issue abstention decisions."""

from __future__ import annotations

import numpy as np
import pytest

from repotriage.inference.abstention import decide_abstention


def test_no_labels_predicted_forces_abstention() -> None:
    result = decide_abstention(
        y_score=np.array([0.1, 0.2], dtype=np.float64),
        y_pred=np.array([0, 0], dtype=np.int8),
        classification_threshold=0.39,
        abstention_threshold=0.84,
        abstention_threshold_basis_points=84,
    )
    assert result.confidence is None
    assert result.should_abstain is True
    assert result.reason == "no_labels_predicted"


def test_confidence_meets_threshold() -> None:
    result = decide_abstention(
        y_score=np.array([0.9, 0.4], dtype=np.float64),
        y_pred=np.array([1, 0], dtype=np.int8),
        classification_threshold=0.39,
        abstention_threshold=0.84,
        abstention_threshold_basis_points=84,
    )
    assert result.confidence == pytest.approx(0.9)
    assert result.should_abstain is False
    assert result.reason == "confidence_meets_threshold"


def test_confidence_below_threshold() -> None:
    result = decide_abstention(
        y_score=np.array([0.6, 0.2], dtype=np.float64),
        y_pred=np.array([1, 0], dtype=np.int8),
        classification_threshold=0.39,
        abstention_threshold=0.84,
        abstention_threshold_basis_points=84,
    )
    assert result.confidence == pytest.approx(0.6)
    assert result.should_abstain is True
    assert result.reason == "confidence_below_threshold"
