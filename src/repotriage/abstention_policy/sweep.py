"""Abstention-threshold grid sweep on validation issue confidences."""

from __future__ import annotations

import numpy as np

from repotriage.abstention_policy.confidence import (
    IssueConfidenceTable,
    build_issue_confidence_table,
    handled_mask_for_abstention_threshold,
)
from repotriage.abstention_policy.evaluator import evaluate_handled_subset
from repotriage.abstention_policy.models import (
    CONFIDENCE_DEFINITION_MAX_PREDICTED,
    AbstentionSweepRow,
    empty_handled_metrics,
)
from repotriage.threshold_policy.models import ThresholdGridConfig


def build_abstention_grid(*, grid: ThresholdGridConfig) -> list[int]:
    """Return ascending basis-point abstention thresholds for the configured grid."""
    return grid.basis_points()


def build_abstention_sweep(
    *,
    labels: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    issue_ids: list[int],
    classification_threshold: float,
    grid: ThresholdGridConfig,
    confidence_definition: str = CONFIDENCE_DEFINITION_MAX_PREDICTED,
) -> tuple[IssueConfidenceTable, list[AbstentionSweepRow]]:
    """Compute validation handled metrics for every abstention threshold in the grid."""
    table = build_issue_confidence_table(
        issue_ids=issue_ids,
        y_score=y_score,
        classification_threshold=classification_threshold,
        confidence_definition=confidence_definition,
    )
    total_count = len(issue_ids)
    forced_abstention_count = int(table.forced_abstention_mask.sum())
    rows: list[AbstentionSweepRow] = []
    for basis_points in build_abstention_grid(grid=grid):
        abstention_threshold = ThresholdGridConfig.threshold_from_basis_points(
            basis_points, grid.denominator
        )
        handled_mask = handled_mask_for_abstention_threshold(table, abstention_threshold)
        handled_count = int(handled_mask.sum())
        abstained_count = total_count - handled_count
        coverage = handled_count / total_count if total_count else 0.0
        abstention_rate = abstained_count / total_count if total_count else 1.0
        if handled_count == 0:
            handled_metrics = empty_handled_metrics()
        else:
            handled_metrics = evaluate_handled_subset(
                labels=labels,
                y_true=y_true[handled_mask],
                y_pred=table.y_pred[handled_mask],
                y_score=y_score[handled_mask],
                classification_threshold=classification_threshold,
            )
        rows.append(
            AbstentionSweepRow(
                abstention_basis_points=basis_points,
                abstention_threshold=abstention_threshold,
                total_count=total_count,
                handled_count=handled_count,
                abstained_count=abstained_count,
                forced_abstention_count=forced_abstention_count,
                coverage=coverage,
                abstention_rate=abstention_rate,
                handled_metrics=handled_metrics,
            )
        )
    return table, rows
