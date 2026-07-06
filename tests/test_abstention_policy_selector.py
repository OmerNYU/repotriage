"""Tests for validation-only abstention selection."""

from __future__ import annotations

import inspect

import pytest

from repotriage.abstention_policy.models import (
    AbstentionPolicyBuildError,
    AbstentionSweepRow,
    HandledMetrics,
)
from repotriage.abstention_policy.selector import select_abstention_threshold


def _metrics(*, subset_accuracy: float, samples_f1: float) -> HandledMetrics:
    return HandledMetrics(
        subset_accuracy=subset_accuracy,
        samples_f1=samples_f1,
        micro_precision=subset_accuracy,
        micro_recall=subset_accuracy,
        micro_f1=subset_accuracy,
        macro_precision=subset_accuracy,
        macro_recall=subset_accuracy,
        macro_f1=subset_accuracy,
        mean_predicted_label_cardinality=1.0,
        mean_true_label_cardinality=1.0,
        false_positive_count=1,
        false_negative_count=1,
    )


def _row(
    basis_points: int,
    *,
    subset_accuracy: float,
    samples_f1: float,
    coverage: float,
    handled_count: int = 10,
) -> AbstentionSweepRow:
    total = 100
    handled = handled_count if handled_count is not None else int(coverage * total)
    abstained = total - handled
    return AbstentionSweepRow(
        abstention_basis_points=basis_points,
        abstention_threshold=basis_points / 100,
        total_count=total,
        handled_count=handled,
        abstained_count=abstained,
        forced_abstention_count=0,
        coverage=coverage,
        abstention_rate=abstained / total,
        handled_metrics=_metrics(subset_accuracy=subset_accuracy, samples_f1=samples_f1),
    )


def test_unique_subset_accuracy_winner() -> None:
    sweep = [
        _row(39, subset_accuracy=0.40, samples_f1=0.50, coverage=0.80),
        _row(50, subset_accuracy=0.55, samples_f1=0.60, coverage=0.70),
    ]
    result = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    assert result.selected_abstention_basis_points == 50


def test_subset_accuracy_tie_resolved_by_samples_f1() -> None:
    sweep = [
        _row(39, subset_accuracy=0.50, samples_f1=0.55, coverage=0.80),
        _row(50, subset_accuracy=0.50, samples_f1=0.70, coverage=0.70),
    ]
    result = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    assert result.selected_abstention_basis_points == 50
    assert "subset_accuracy_tied" in result.selection_audit.tie_break_steps


def test_tie_resolved_by_coverage_then_lower_threshold() -> None:
    sweep = [
        _row(60, subset_accuracy=0.50, samples_f1=0.60, coverage=0.80),
        _row(70, subset_accuracy=0.50, samples_f1=0.60, coverage=0.70),
        _row(65, subset_accuracy=0.50, samples_f1=0.60, coverage=0.70),
    ]
    result = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    assert result.selected_abstention_basis_points == 60


def test_minimum_coverage_floor_enforced() -> None:
    sweep = [
        _row(95, subset_accuracy=0.90, samples_f1=0.90, coverage=0.10),
        _row(50, subset_accuracy=0.45, samples_f1=0.50, coverage=0.30),
    ]
    result = select_abstention_threshold(
        sweep=sweep,
        minimum_coverage=0.25,
        classification_threshold_basis_points=39,
    )
    assert result.selected_abstention_basis_points == 50


def test_no_eligible_rows_fails_loudly() -> None:
    sweep = [
        _row(95, subset_accuracy=0.90, samples_f1=0.90, coverage=0.10),
    ]
    with pytest.raises(AbstentionPolicyBuildError, match="minimum coverage"):
        select_abstention_threshold(
            sweep=sweep,
            minimum_coverage=0.25,
            classification_threshold_basis_points=39,
        )


def test_selector_has_no_test_parameters() -> None:
    signature = inspect.signature(select_abstention_threshold)
    for name in signature.parameters:
        assert "test" not in name.lower()
