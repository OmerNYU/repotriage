"""Target-label vector construction from canonical policy order."""

from __future__ import annotations

from repotriage.model_dataset.models import ModelDatasetTransformError


def build_target_labels(
    issue_labels: list[str],
    canonical_order: list[str],
) -> tuple[list[str], list[int]]:
    """Build selected labels and a binary target vector in canonical order.

    ``selected_labels`` contains only labels present on the issue, ordered by
    ``canonical_order``. ``target_vector`` has one entry per canonical label.
    """
    if not canonical_order:
        raise ModelDatasetTransformError("canonical_order must not be empty")
    if len(set(canonical_order)) != len(canonical_order):
        raise ModelDatasetTransformError("canonical_order must not contain duplicates")

    issue_label_set = set(issue_labels)
    vector = [1 if label in issue_label_set else 0 for label in canonical_order]
    selected = [label for label in canonical_order if label in issue_label_set]
    return selected, vector


def assert_canonical_order_matches_policy(
    canonical_order: list[str],
    include_decision_labels: list[str],
) -> None:
    """Assert policy included_labels agrees exactly with ordered include decisions."""
    if canonical_order != include_decision_labels:
        raise ModelDatasetTransformError(
            "coverage.included_labels does not match the ordered include decisions"
        )
