"""Validate inference responses."""

from __future__ import annotations

from repotriage.inference.models import InferenceResponse


def validate_inference_response(response: InferenceResponse) -> None:
    """Validate structural invariants on a combined inference response."""
    label_order = response.classification.label_order
    if len(response.classification.scores) != len(label_order):
        raise ValueError("classification.scores length does not match label_order.")
    for index, entry in enumerate(response.classification.scores):
        if entry.label != label_order[index]:
            raise ValueError(
                f"classification.scores[{index}] label {entry.label!r} "
                f"does not match label_order[{index}] {label_order[index]!r}."
            )

    for item in response.classification.predicted_labels:
        if item.score < response.classification.threshold:
            raise ValueError(
                f"predicted label {item.label!r} score {item.score} is below threshold."
            )

    if not response.classification.predicted_labels:
        if response.abstention.reason != "no_labels_predicted":
            raise ValueError(
                "abstention.reason must be no_labels_predicted when predicted_labels is empty."
            )
        if response.abstention.confidence is not None:
            raise ValueError("abstention confidence must be null when no labels are predicted.")
        if not response.abstention.should_abstain:
            raise ValueError("abstention.should_abstain must be true when no labels are predicted.")

    ranks = [neighbor.rank for neighbor in response.retrieval.similar_issues]
    if ranks and ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("retrieval similar_issues ranks are not contiguous from 1.")

    previous_similarity = float("inf")
    for neighbor in response.retrieval.similar_issues:
        if neighbor.similarity > previous_similarity + 1e-12:
            raise ValueError("retrieval similar_issues similarities are not descending.")
        previous_similarity = neighbor.similarity
