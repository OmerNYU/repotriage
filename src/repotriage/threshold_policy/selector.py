"""Validation-only global threshold selection with deterministic tie-breaks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from repotriage.baseline.models import SplitMetrics, floats_consistent
from repotriage.threshold_policy.models import (
    THRESHOLD_SELECTION_RULE_VERSION,
    ThresholdSelectionAudit,
    ThresholdSweepRow,
)


@dataclass(frozen=True)
class ThresholdSelectionResult:
    selected_threshold_basis_points: int
    selected_threshold: float
    selected_validation_metrics: SplitMetrics
    selection_audit: ThresholdSelectionAudit


@dataclass(frozen=True)
class FrozenThresholdPolicy:
    selected_threshold_basis_points: int
    selected_threshold: float
    selected_validation_metrics: SplitMetrics
    selection_audit: ThresholdSelectionAudit


def _metric_value(metrics: SplitMetrics, name: str) -> float:
    value = getattr(metrics.aggregate, name)
    if value is None:
        return float("-inf")
    return float(value)


def _distance_to_reference_basis_points(
    basis_points: int, reference_basis_points: int
) -> int:
    return abs(basis_points - reference_basis_points)


def select_threshold(
    *,
    sweep: list[ThresholdSweepRow],
    reference_threshold_basis_points: int = 50,
    selection_rule_version: Literal["1"] = THRESHOLD_SELECTION_RULE_VERSION,
    denominator: int = 100,
) -> ThresholdSelectionResult:
    """Select one global threshold using validation metrics only."""
    if not sweep:
        raise ValueError("sweep must not be empty")
    if selection_rule_version != "1":
        raise ValueError(f"Unsupported selection_rule_version: {selection_rule_version!r}")

    ranked = sorted(
        sweep,
        key=lambda row: (
            _metric_value(row.metrics, "macro_f1"),
            _metric_value(row.metrics, "micro_f1"),
            -_distance_to_reference_basis_points(
                row.threshold_basis_points, reference_threshold_basis_points
            ),
            row.threshold_basis_points,
        ),
        reverse=True,
    )

    winner = ranked[0]
    tie_break_steps: list[str] = []
    if len(ranked) > 1:
        runner_up = ranked[1]
        if floats_consistent(
            _metric_value(winner.metrics, "macro_f1"),
            _metric_value(runner_up.metrics, "macro_f1"),
        ):
            tie_break_steps.append("macro_f1_tied")
        if floats_consistent(
            _metric_value(winner.metrics, "micro_f1"),
            _metric_value(runner_up.metrics, "micro_f1"),
        ):
            tie_break_steps.append("micro_f1_tied")
        winner_distance = _distance_to_reference_basis_points(
            winner.threshold_basis_points, reference_threshold_basis_points
        )
        runner_up_distance = _distance_to_reference_basis_points(
            runner_up.threshold_basis_points, reference_threshold_basis_points
        )
        if floats_consistent(winner_distance, runner_up_distance):
            tie_break_steps.append("distance_to_reference_tied")
        if winner.threshold_basis_points == runner_up.threshold_basis_points:
            tie_break_steps.append("threshold_basis_points_tied")

    audit = ThresholdSelectionAudit(
        selection_rule_version=selection_rule_version,
        selected_threshold_basis_points=winner.threshold_basis_points,
        reference_threshold_basis_points=reference_threshold_basis_points,
        tie_break_steps=tie_break_steps,
        ranked_threshold_basis_points=[row.threshold_basis_points for row in ranked],
    )
    return ThresholdSelectionResult(
        selected_threshold_basis_points=winner.threshold_basis_points,
        selected_threshold=winner.threshold,
        selected_validation_metrics=winner.metrics,
        selection_audit=audit,
    )


def freeze_threshold_policy(selection: ThresholdSelectionResult) -> FrozenThresholdPolicy:
    """Freeze the selected threshold before any test evaluation."""
    return FrozenThresholdPolicy(
        selected_threshold_basis_points=selection.selected_threshold_basis_points,
        selected_threshold=selection.selected_threshold,
        selected_validation_metrics=selection.selected_validation_metrics,
        selection_audit=selection.selection_audit,
    )
