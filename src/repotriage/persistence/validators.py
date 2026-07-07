"""Domain validation for maintainer feedback requests."""

from __future__ import annotations

import re

from repotriage.abstention_policy.models import POLICY_ID_PATTERN as ABSTENTION_POLICY_ID_PATTERN
from repotriage.baseline.models import BASELINE_RUN_ID_PATTERN
from repotriage.github.models import InvalidRepositoryError, parse_repository
from repotriage.inference.artifact_loader import LoadedInferenceBundle
from repotriage.model_dataset.models import MODEL_DATASET_ID_PATTERN
from repotriage.persistence.errors import FeedbackValidationError
from repotriage.persistence.schemas import FeedbackRequest
from repotriage.retrieval.models import RETRIEVAL_RUN_ID_PATTERN
from repotriage.threshold_policy.models import POLICY_ID_PATTERN as THRESHOLD_POLICY_ID_PATTERN

_MODEL_DATASET_ID_RE = re.compile(MODEL_DATASET_ID_PATTERN)
_BASELINE_RUN_ID_RE = re.compile(BASELINE_RUN_ID_PATTERN)
_THRESHOLD_POLICY_ID_RE = re.compile(THRESHOLD_POLICY_ID_PATTERN)
_ABSTENTION_POLICY_ID_RE = re.compile(ABSTENTION_POLICY_ID_PATTERN)
_RETRIEVAL_RUN_ID_RE = re.compile(RETRIEVAL_RUN_ID_PATTERN)


def validate_feedback_request(body: FeedbackRequest, bundle: LoadedInferenceBundle) -> None:
    """Validate a feedback request against the loaded inference bundle."""
    _validate_repository(body.repository, bundle)
    _validate_artifact_ids(body, bundle)
    _validate_labels(body, bundle.label_order)
    _validate_review_action(body)


def _validate_repository(repository: str, bundle: LoadedInferenceBundle) -> None:
    try:
        parse_repository(repository)
    except InvalidRepositoryError as exc:
        raise FeedbackValidationError(str(exc)) from exc

    if repository != bundle.repository.full_name:
        raise FeedbackValidationError(
            f"Repository {repository!r} does not match server-bound "
            f"{bundle.repository.full_name!r}."
        )


def _validate_artifact_ids(body: FeedbackRequest, bundle: LoadedInferenceBundle) -> None:
    artifacts = body.inference_artifacts
    checks = (
        ("model_dataset_id", artifacts.model_dataset_id, _MODEL_DATASET_ID_RE),
        ("baseline_run_id", artifacts.baseline_run_id, _BASELINE_RUN_ID_RE),
        ("threshold_policy_id", artifacts.threshold_policy_id, _THRESHOLD_POLICY_ID_RE),
        ("abstention_policy_id", artifacts.abstention_policy_id, _ABSTENTION_POLICY_ID_RE),
        ("retrieval_run_id", artifacts.retrieval_run_id, _RETRIEVAL_RUN_ID_RE),
    )
    config = bundle.config
    expected = {
        "model_dataset_id": config.model_dataset_id,
        "baseline_run_id": config.baseline_run_id,
        "threshold_policy_id": config.threshold_policy_id,
        "abstention_policy_id": config.abstention_policy_id,
        "retrieval_run_id": config.retrieval_run_id,
    }

    for field_name, value, pattern in checks:
        if not pattern.fullmatch(value):
            raise FeedbackValidationError(f"Invalid {field_name} format: {value!r}.")
        if value != expected[field_name]:
            raise FeedbackValidationError(
                f"{field_name} {value!r} does not match loaded inference bundle "
                f"{expected[field_name]!r}."
            )


def _validate_labels(body: FeedbackRequest, label_order: list[str]) -> None:
    allowed = set(label_order)
    for field_name, labels in (
        ("predicted_labels", body.predicted_labels),
        ("accepted_labels", body.accepted_labels),
        ("rejected_labels", body.rejected_labels),
    ):
        _validate_label_list(field_name, labels, allowed)

    accepted_set = set(body.accepted_labels)
    rejected_set = set(body.rejected_labels)
    overlap = accepted_set & rejected_set
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise FeedbackValidationError(
            f"accepted_labels and rejected_labels must be disjoint; overlap: {joined}."
        )


def _validate_label_list(field_name: str, labels: list[str], allowed: set[str]) -> None:
    seen: set[str] = set()
    for label in labels:
        if label not in allowed:
            raise FeedbackValidationError(f"Unknown label in {field_name}: {label!r}.")
        if label in seen:
            raise FeedbackValidationError(f"Duplicate label in {field_name}: {label!r}.")
        seen.add(label)


def _validate_review_action(body: FeedbackRequest) -> None:
    predicted = body.predicted_labels
    accepted = body.accepted_labels
    rejected = body.rejected_labels

    if body.review_action == "accepted":
        if accepted != predicted:
            raise FeedbackValidationError(
                "review_action 'accepted' requires accepted_labels to equal predicted_labels."
            )
        if rejected:
            raise FeedbackValidationError(
                "review_action 'accepted' requires rejected_labels to be empty."
            )
        return

    if body.review_action == "corrected":
        if accepted == predicted:
            raise FeedbackValidationError(
                "review_action 'corrected' requires accepted_labels to differ from "
                "predicted_labels."
            )
        return

    if body.review_action == "rejected":
        if accepted:
            raise FeedbackValidationError(
                "review_action 'rejected' requires accepted_labels to be empty."
            )
        if set(rejected) != set(predicted):
            raise FeedbackValidationError(
                "review_action 'rejected' requires rejected_labels to equal predicted_labels."
            )
