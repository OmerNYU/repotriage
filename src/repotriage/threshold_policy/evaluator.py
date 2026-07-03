"""Evaluate a frozen global threshold on held-out test scores."""

from __future__ import annotations

from repotriage.baseline.evaluator import compute_split_metrics
from repotriage.baseline.models import SplitMetrics
from repotriage.threshold_policy.reader import TestScoreBundle
from repotriage.threshold_policy.selector import FrozenThresholdPolicy
from repotriage.threshold_policy.sweep import predictions_from_scores


def evaluate_frozen_threshold(
    *,
    frozen: FrozenThresholdPolicy,
    bundle: TestScoreBundle,
) -> SplitMetrics:
    """Evaluate the frozen global threshold on test scores."""
    y_pred = predictions_from_scores(bundle.y_score, frozen.selected_threshold)
    return compute_split_metrics(
        split="test",
        labels=bundle.labels,
        y_true=bundle.y_true,
        y_pred=y_pred,
        y_score=bundle.y_score,
        threshold=frozen.selected_threshold,
        score_type="probability_estimates",
    )
