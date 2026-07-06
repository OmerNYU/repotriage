"""Validation-only abstention threshold selection with deterministic tie-breaks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from repotriage.abstention_policy.models import (
    ABSTENTION_SELECTION_RULE_VERSION,
    AbstentionPolicyBuildError,
    AbstentionSelectionAudit,
    AbstentionSweepRow,
    HandledMetrics,
)
from repotriage.baseline.models import floats_consistent


@dataclass(frozen=True)
class AbstentionSelectionResult:
    selected_abstention_basis_points: int
    selected_abstention_threshold: float
    selected_validation_metrics: HandledMetrics
    selected_validation_coverage: float
    selected_validation_handled_count: int
    selection_audit: AbstentionSelectionAudit


@dataclass(frozen=True)
class FrozenAbstentionPolicy:
    selected_abstention_basis_points: int
    selected_abstention_threshold: float
    classification_threshold_basis_points: int
    classification_threshold: float
    selected_validation_metrics: HandledMetrics
    selected_validation_coverage: float
    selected_validation_handled_count: int
    selection_audit: AbstentionSelectionAudit


def _metric_value(row: AbstentionSweepRow, name: str) -> float:
    value = getattr(row.handled_metrics, name)
    if value is None:
        return float("-inf")
    return float(value)


def select_abstention_threshold(
    *,
    sweep: list[AbstentionSweepRow],
    minimum_coverage: float,
    classification_threshold_basis_points: int,
    selection_rule_version: Literal["1"] = ABSTENTION_SELECTION_RULE_VERSION,
    denominator: int = 100,
) -> AbstentionSelectionResult:
    """Select one abstention threshold using validation handled metrics only."""
    if not sweep:
        raise ValueError("sweep must not be empty")
    if selection_rule_version != "1":
        raise ValueError(f"Unsupported selection_rule_version: {selection_rule_version!r}")

    eligible = [row for row in sweep if row.coverage >= minimum_coverage]
    if not eligible:
        raise AbstentionPolicyBuildError(
            f"No abstention sweep row met minimum coverage {minimum_coverage:.6f}."
        )

    ranked = sorted(
        eligible,
        key=lambda row: (
            _metric_value(row, "subset_accuracy"),
            _metric_value(row, "samples_f1"),
            row.coverage,
            -row.abstention_basis_points,
        ),
        reverse=True,
    )

    winner = ranked[0]
    tie_break_steps: list[str] = []
    if len(ranked) > 1:
        runner_up = ranked[1]
        if floats_consistent(
            _metric_value(winner, "subset_accuracy"),
            _metric_value(runner_up, "subset_accuracy"),
        ):
            tie_break_steps.append("subset_accuracy_tied")
        if floats_consistent(
            _metric_value(winner, "samples_f1"),
            _metric_value(runner_up, "samples_f1"),
        ):
            tie_break_steps.append("samples_f1_tied")
        if floats_consistent(winner.coverage, runner_up.coverage):
            tie_break_steps.append("coverage_tied")
        if winner.abstention_basis_points == runner_up.abstention_basis_points:
            tie_break_steps.append("abstention_basis_points_tied")

    audit = AbstentionSelectionAudit(
        selection_rule_version=selection_rule_version,
        classification_threshold_basis_points=classification_threshold_basis_points,
        selected_abstention_basis_points=winner.abstention_basis_points,
        minimum_coverage=minimum_coverage,
        tie_break_steps=tie_break_steps,
        ranked_abstention_basis_points=[row.abstention_basis_points for row in ranked],
    )
    return AbstentionSelectionResult(
        selected_abstention_basis_points=winner.abstention_basis_points,
        selected_abstention_threshold=winner.abstention_threshold,
        selected_validation_metrics=winner.handled_metrics,
        selected_validation_coverage=winner.coverage,
        selected_validation_handled_count=winner.handled_count,
        selection_audit=audit,
    )


def freeze_abstention_policy(
    *,
    selection: AbstentionSelectionResult,
    classification_threshold_basis_points: int,
    denominator: int,
) -> FrozenAbstentionPolicy:
    """Freeze the selected abstention threshold before any test evaluation."""
    return FrozenAbstentionPolicy(
        selected_abstention_basis_points=selection.selected_abstention_basis_points,
        selected_abstention_threshold=selection.selected_abstention_threshold,
        classification_threshold_basis_points=classification_threshold_basis_points,
        classification_threshold=classification_threshold_basis_points / denominator,
        selected_validation_metrics=selection.selected_validation_metrics,
        selected_validation_coverage=selection.selected_validation_coverage,
        selected_validation_handled_count=selection.selected_validation_handled_count,
        selection_audit=selection.selection_audit,
    )
