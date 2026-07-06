"""Tests for abstention-threshold sweep."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from repotriage.abstention_policy.sweep import build_abstention_sweep
from repotriage.threshold_policy.models import ThresholdGridConfig


def test_sweep_counts_and_rates() -> None:
    labels = ["Bug", "Docs"]
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    y_score = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.1, 0.1],
        ],
        dtype=np.float64,
    )
    grid = ThresholdGridConfig(
        start_basis_points=39,
        stop_basis_points=41,
        step_basis_points=1,
        denominator=100,
    )
    _, rows = build_abstention_sweep(
        labels=labels,
        y_true=y_true,
        y_score=y_score,
        issue_ids=[1, 2, 3],
        classification_threshold=0.39,
        grid=grid,
    )
    low_row = rows[0]
    assert low_row.total_count == 3
    assert low_row.forced_abstention_count == 1
    assert low_row.handled_count == 2
    assert low_row.abstained_count == 1
    assert low_row.coverage == pytest.approx(2 / 3)
    assert low_row.abstention_rate == pytest.approx(1 / 3)


def test_handled_count_zero_metrics_are_null() -> None:
    labels = ["Bug"]
    y_true = np.array([[1]], dtype=np.int8)
    y_score = np.array([[0.1]], dtype=np.float64)
    grid = ThresholdGridConfig(
        start_basis_points=39,
        stop_basis_points=95,
        step_basis_points=1,
        denominator=100,
    )
    _, rows = build_abstention_sweep(
        labels=labels,
        y_true=y_true,
        y_score=y_score,
        issue_ids=[1],
        classification_threshold=0.39,
        grid=grid,
    )
    for row in rows:
        assert row.handled_count == 0
        assert row.coverage == 0.0
        assert row.abstention_rate == 1.0
        assert row.handled_metrics.subset_accuracy is None


def test_build_abstention_sweep_has_no_test_parameters() -> None:
    signature = inspect.signature(build_abstention_sweep)
    for name in signature.parameters:
        assert "test" not in name.lower()
