"""Handled-subset evaluation and confidence-bin diagnostics."""

from __future__ import annotations

import numpy as np

from repotriage.abstention_policy.confidence import (
    IssueConfidenceTable,
    build_issue_confidence_table,
    handled_mask_for_abstention_threshold,
)
from repotriage.abstention_policy.models import (
    CONFIDENCE_DEFINITION_MAX_PREDICTED,
    AbstentionSplitMetrics,
    ConfidenceBinRow,
    ConfidenceBinsDocument,
    HandledMetrics,
    NoPredictionBucket,
    handled_metrics_from_split_metrics,
)
from repotriage.baseline.evaluator import compute_split_metrics
from repotriage.model_dataset.models import SplitName

CONFIDENCE_BIN_SPECS: list[tuple[str, float, float]] = [
    ("0.39-0.49", 0.39, 0.50),
    ("0.50-0.59", 0.50, 0.60),
    ("0.60-0.69", 0.60, 0.70),
    ("0.70-0.79", 0.70, 0.80),
    ("0.80-0.89", 0.80, 0.90),
    ("0.90-1.00", 0.90, 1.00),
]


def evaluate_handled_subset(
    *,
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float,
) -> HandledMetrics:
    """Compute handled-subset metrics by reusing the baseline metric contract."""
    metrics = compute_split_metrics(
        split="validation",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        threshold=classification_threshold,
        score_type="probability_estimates",
    )
    return handled_metrics_from_split_metrics(metrics)


def evaluate_frozen_abstention_split(
    *,
    split: SplitName,
    labels: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    table: IssueConfidenceTable,
    classification_threshold: float,
    abstention_threshold: float,
    full_set_reference: HandledMetrics | None = None,
) -> AbstentionSplitMetrics:
    """Evaluate one split at a frozen abstention threshold."""
    total_count = len(table.confidences)
    handled_mask = handled_mask_for_abstention_threshold(table, abstention_threshold)
    handled_count = int(handled_mask.sum())
    abstained_count = total_count - handled_count
    forced_abstention_count = int(table.forced_abstention_mask.sum())
    coverage = handled_count / total_count if total_count else 0.0
    abstention_rate = abstained_count / total_count if total_count else 1.0
    if handled_count == 0:
        handled_metrics = HandledMetrics()
    else:
        handled_metrics = evaluate_handled_subset(
            labels=labels,
            y_true=y_true[handled_mask],
            y_pred=table.y_pred[handled_mask],
            y_score=y_score[handled_mask],
            classification_threshold=classification_threshold,
        )
    return AbstentionSplitMetrics(
        split=split,
        classification_threshold=classification_threshold,
        abstention_threshold=abstention_threshold,
        total_count=total_count,
        handled_count=handled_count,
        abstained_count=abstained_count,
        forced_abstention_count=forced_abstention_count,
        coverage=coverage,
        abstention_rate=abstention_rate,
        handled_metrics=handled_metrics,
        full_set_reference=full_set_reference,
    )


def _confidence_in_bin(confidence: float, lower: float, upper: float, *, last_bin: bool) -> bool:
    if last_bin:
        return lower <= confidence <= upper
    return lower <= confidence < upper


def compute_confidence_bins(
    *,
    split: SplitName,
    labels: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    issue_ids: list[int],
    classification_threshold: float,
    confidence_definition: str = CONFIDENCE_DEFINITION_MAX_PREDICTED,
) -> ConfidenceBinsDocument:
    """Compute confidence-bin diagnostics for issues with predicted labels."""
    table = build_issue_confidence_table(
        issue_ids=issue_ids,
        y_score=y_score,
        classification_threshold=classification_threshold,
        confidence_definition=confidence_definition,
    )
    total_count = len(issue_ids)
    no_prediction_count = int(table.forced_abstention_mask.sum())
    bins: list[ConfidenceBinRow] = []
    for index, (label, lower, upper) in enumerate(CONFIDENCE_BIN_SPECS):
        last_bin = index == len(CONFIDENCE_BIN_SPECS) - 1
        bin_mask = np.zeros(total_count, dtype=bool)
        for row_index, confidence in enumerate(table.confidences):
            if confidence is None:
                continue
            if _confidence_in_bin(confidence, lower, upper, last_bin=last_bin):
                bin_mask[row_index] = True
        issue_count = int(bin_mask.sum())
        if issue_count == 0:
            bins.append(
                ConfidenceBinRow(
                    bin_label=label,
                    lower_bound=lower,
                    upper_bound=upper,
                    issue_count=0,
                    fraction_of_all_issues=0.0,
                    subset_accuracy=None,
                    samples_f1=None,
                    mean_predicted_label_cardinality=None,
                    mean_true_label_cardinality=None,
                )
            )
            continue
        metrics = evaluate_handled_subset(
            labels=labels,
            y_true=y_true[bin_mask],
            y_pred=table.y_pred[bin_mask],
            y_score=y_score[bin_mask],
            classification_threshold=classification_threshold,
        )
        bins.append(
            ConfidenceBinRow(
                bin_label=label,
                lower_bound=lower,
                upper_bound=upper,
                issue_count=issue_count,
                fraction_of_all_issues=issue_count / total_count if total_count else 0.0,
                subset_accuracy=metrics.subset_accuracy,
                samples_f1=metrics.samples_f1,
                mean_predicted_label_cardinality=metrics.mean_predicted_label_cardinality,
                mean_true_label_cardinality=metrics.mean_true_label_cardinality,
            )
        )
    return ConfidenceBinsDocument(
        split=split,
        classification_threshold=classification_threshold,
        confidence_definition=CONFIDENCE_DEFINITION_MAX_PREDICTED,
        bins=bins,
        no_prediction_bucket=NoPredictionBucket(
            issue_count=no_prediction_count,
            fraction_of_all_issues=no_prediction_count / total_count if total_count else 0.0,
        ),
    )
