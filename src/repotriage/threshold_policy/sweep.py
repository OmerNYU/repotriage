"""Basis-point threshold grid construction and validation metric sweep."""

from __future__ import annotations

import numpy as np

from repotriage.baseline.evaluator import compute_split_metrics
from repotriage.threshold_policy.models import ThresholdGridConfig, ThresholdSweepRow


def build_threshold_grid(*, grid: ThresholdGridConfig) -> list[int]:
    """Return ascending basis-point thresholds for the configured grid."""
    return grid.basis_points()


def predictions_from_scores(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Apply inclusive threshold comparison: score >= threshold predicts positive."""
    return (y_score >= threshold).astype(np.int8)


def build_threshold_sweep(
    *,
    labels: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: ThresholdGridConfig,
) -> list[ThresholdSweepRow]:
    """Compute validation metrics for every threshold in the grid."""
    rows: list[ThresholdSweepRow] = []
    for basis_points in build_threshold_grid(grid=grid):
        threshold = ThresholdGridConfig.threshold_from_basis_points(
            basis_points, grid.denominator
        )
        y_pred = predictions_from_scores(y_score, threshold)
        metrics = compute_split_metrics(
            split="validation",
            labels=labels,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            threshold=threshold,
            score_type="probability_estimates",
        )
        rows.append(
            ThresholdSweepRow(
                threshold_basis_points=basis_points,
                threshold=threshold,
                metrics=metrics,
            )
        )
    return rows
