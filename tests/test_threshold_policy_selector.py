"""Tests for validation-only threshold selection."""

from __future__ import annotations

import inspect

from repotriage.baseline.models import (
    AggregateMetrics,
    PerLabelMetric,
    SplitMetrics,
    floats_consistent,
)
from repotriage.threshold_policy.models import ThresholdSweepRow
from repotriage.threshold_policy.selector import select_threshold


def _aggregate(*, macro_f1: float, micro_f1: float) -> AggregateMetrics:
    return AggregateMetrics(
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        macro_precision=macro_f1,
        macro_recall=macro_f1,
        macro_label_count=1,
        macro_precision_denominator=1,
        macro_recall_denominator=1,
        macro_f1_denominator=1,
        undefined_precision_label_count=0,
        undefined_recall_label_count=0,
        undefined_f1_label_count=0,
        macro_average_precision_label_count=0,
        subset_accuracy=0.5,
        hamming_loss=0.1,
        mean_true_label_cardinality=1.0,
        mean_predicted_label_cardinality=1.0,
        fraction_no_prediction=0.0,
        fraction_any_prediction=1.0,
        record_count=10,
    )


def _row(basis_points: int, *, macro_f1: float, micro_f1: float) -> ThresholdSweepRow:
    threshold = basis_points / 100
    return ThresholdSweepRow(
        threshold_basis_points=basis_points,
        threshold=threshold,
        metrics=SplitMetrics(
            split="validation",
            threshold=threshold,
            score_type="probability_estimates",
            per_label=[
                PerLabelMetric(
                    label="Bug",
                    support=5,
                    prevalence=0.5,
                    predicted_positives=3,
                    tp=2,
                    fp=1,
                    fn=3,
                    tn=4,
                )
            ],
            aggregate=_aggregate(macro_f1=macro_f1, micro_f1=micro_f1),
        ),
    )


def test_unique_macro_f1_winner() -> None:
    sweep = [
        _row(30, macro_f1=0.40, micro_f1=0.50),
        _row(39, macro_f1=0.52, micro_f1=0.60),
        _row(50, macro_f1=0.47, micro_f1=0.66),
    ]
    result = select_threshold(sweep=sweep)
    assert result.selected_threshold_basis_points == 39


def test_macro_f1_tie_resolved_by_micro_f1() -> None:
    sweep = [
        _row(30, macro_f1=0.50, micro_f1=0.55),
        _row(40, macro_f1=0.50, micro_f1=0.70),
    ]
    result = select_threshold(sweep=sweep)
    assert result.selected_threshold_basis_points == 40
    assert "macro_f1_tied" in result.selection_audit.tie_break_steps


def test_tie_resolved_by_distance_to_reference() -> None:
    sweep = [
        _row(48, macro_f1=0.50, micro_f1=0.60),
        _row(51, macro_f1=0.50, micro_f1=0.60),
    ]
    result = select_threshold(sweep=sweep)
    assert result.selected_threshold_basis_points == 51


def test_equal_distance_tie_resolved_by_higher_threshold() -> None:
    sweep = [
        _row(45, macro_f1=0.50, micro_f1=0.60),
        _row(55, macro_f1=0.50, micro_f1=0.60),
    ]
    result = select_threshold(sweep=sweep)
    assert result.selected_threshold_basis_points == 55
    assert "distance_to_reference_tied" in result.selection_audit.tie_break_steps


def test_floats_consistent_at_tolerance_boundary() -> None:
    assert floats_consistent(0.5, 0.5 + 1e-13)
    assert not floats_consistent(0.5, 0.5 + 1e-11)


def test_selector_has_no_test_parameters() -> None:
    signature = inspect.signature(select_threshold)
    for name in signature.parameters:
        assert "test" not in name.lower()
