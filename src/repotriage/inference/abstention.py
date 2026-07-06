"""Single-issue abstention decision using frozen abstention policy."""

from __future__ import annotations

import numpy as np

from repotriage.abstention_policy.confidence import issue_confidence_from_scores
from repotriage.abstention_policy.models import CONFIDENCE_DEFINITION_MAX_PREDICTED
from repotriage.inference.models import AbstentionResult


def decide_abstention(
    *,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    classification_threshold: float,
    abstention_threshold: float,
    abstention_threshold_basis_points: int,
    confidence_definition: str = CONFIDENCE_DEFINITION_MAX_PREDICTED,
) -> AbstentionResult:
    """Return one abstention decision for a single issue."""
    confidence = issue_confidence_from_scores(
        y_score,
        y_pred,
        confidence_definition=confidence_definition,
    )
    if confidence is None:
        return AbstentionResult(
            confidence=None,
            threshold=abstention_threshold,
            threshold_basis_points=abstention_threshold_basis_points,
            should_abstain=True,
            reason="no_labels_predicted",
        )
    if confidence >= abstention_threshold:
        return AbstentionResult(
            confidence=confidence,
            threshold=abstention_threshold,
            threshold_basis_points=abstention_threshold_basis_points,
            should_abstain=False,
            reason="confidence_meets_threshold",
        )
    return AbstentionResult(
        confidence=confidence,
        threshold=abstention_threshold,
        threshold_basis_points=abstention_threshold_basis_points,
        should_abstain=True,
        reason="confidence_below_threshold",
    )
