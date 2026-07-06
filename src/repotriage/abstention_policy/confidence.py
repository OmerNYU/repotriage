"""Issue-level confidence computation from frozen classification predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repotriage.abstention_policy.models import CONFIDENCE_DEFINITION_MAX_PREDICTED


@dataclass(frozen=True)
class IssueConfidenceTable:
    """Per-issue classification predictions and confidence."""

    issue_ids: list[int]
    y_pred: np.ndarray
    confidences: list[float | None]
    forced_abstention_mask: np.ndarray


def issue_confidence_from_scores(
    y_score_row: np.ndarray,
    y_pred_row: np.ndarray,
    *,
    confidence_definition: str = CONFIDENCE_DEFINITION_MAX_PREDICTED,
) -> float | None:
    """Return issue confidence or None when no labels are predicted."""
    if confidence_definition != CONFIDENCE_DEFINITION_MAX_PREDICTED:
        raise ValueError(f"Unsupported confidence_definition: {confidence_definition!r}")
    if not y_pred_row.any():
        return None
    return float(y_score_row[y_pred_row.astype(bool)].max())


def build_issue_confidence_table(
    *,
    issue_ids: list[int],
    y_score: np.ndarray,
    classification_threshold: float,
    confidence_definition: str = CONFIDENCE_DEFINITION_MAX_PREDICTED,
) -> IssueConfidenceTable:
    """Derive per-issue predictions and confidence at the classification threshold."""
    y_pred = (y_score >= classification_threshold).astype(np.int8)
    confidences: list[float | None] = []
    forced_mask = np.zeros(y_score.shape[0], dtype=bool)
    for row_index in range(y_score.shape[0]):
        row_pred = y_pred[row_index]
        if not row_pred.any():
            forced_mask[row_index] = True
            confidences.append(None)
            continue
        confidences.append(
            issue_confidence_from_scores(
                y_score[row_index],
                row_pred,
                confidence_definition=confidence_definition,
            )
        )
    return IssueConfidenceTable(
        issue_ids=issue_ids,
        y_pred=y_pred,
        confidences=confidences,
        forced_abstention_mask=forced_mask,
    )


def handled_mask_for_abstention_threshold(
    table: IssueConfidenceTable,
    abstention_threshold: float,
) -> np.ndarray:
    """Return a boolean mask of handled issues for one abstention threshold."""
    mask = np.zeros(len(table.confidences), dtype=bool)
    for index, confidence in enumerate(table.confidences):
        if confidence is None:
            continue
        if confidence >= abstention_threshold:
            mask[index] = True
    return mask
