"""Tests for feedback request validators."""

from __future__ import annotations

import pytest

from repotriage.persistence.errors import FeedbackValidationError
from repotriage.persistence.schemas import FeedbackRequest, InferenceArtifactsInput
from repotriage.persistence.validators import validate_feedback_request
from tests.helpers import (
    TEST_ABSTENTION_POLICY_ID,
    TEST_BASELINE_RUN_ID,
    TEST_MODEL_DATASET_ID,
    TEST_RETRIEVAL_RUN_ID,
    TEST_THRESHOLD_POLICY_ID,
    make_test_bundle,
)


def _request(
    *,
    repository: str = "pandas-dev/pandas",
    predicted_labels: list[str] | None = None,
    accepted_labels: list[str] | None = None,
    rejected_labels: list[str] | None = None,
    review_action: str = "corrected",
    model_dataset_id: str = TEST_MODEL_DATASET_ID,
) -> FeedbackRequest:
    return FeedbackRequest(
        repository=repository,
        issue_number=12345,
        issue_title="BUG: example",
        issue_body_preview="Preview",
        predicted_labels=predicted_labels if predicted_labels is not None else ["Indexing"],
        accepted_labels=accepted_labels if accepted_labels is not None else ["Bug", "Indexing"],
        rejected_labels=rejected_labels if rejected_labels is not None else [],
        review_action=review_action,
        inference_artifacts=InferenceArtifactsInput(
            model_dataset_id=model_dataset_id,
            baseline_run_id=TEST_BASELINE_RUN_ID,
            threshold_policy_id=TEST_THRESHOLD_POLICY_ID,
            abstention_policy_id=TEST_ABSTENTION_POLICY_ID,
            retrieval_run_id=TEST_RETRIEVAL_RUN_ID,
        ),
    )


def test_validate_feedback_request_accepts_valid_corrected_payload() -> None:
    validate_feedback_request(_request(), make_test_bundle())


def test_validate_feedback_request_rejects_repository_mismatch() -> None:
    with pytest.raises(FeedbackValidationError, match="does not match server-bound"):
        validate_feedback_request(_request(repository="other/repo"), make_test_bundle())


def test_validate_feedback_request_rejects_unknown_label() -> None:
    with pytest.raises(FeedbackValidationError, match="Unknown label"):
        validate_feedback_request(
            _request(predicted_labels=["NotARealLabel"]),
            make_test_bundle(),
        )


def test_validate_feedback_request_rejects_duplicate_label() -> None:
    with pytest.raises(FeedbackValidationError, match="Duplicate label"):
        validate_feedback_request(
            _request(predicted_labels=["Bug", "Bug"]),
            make_test_bundle(),
        )


def test_validate_feedback_request_rejects_accepted_rejected_overlap() -> None:
    with pytest.raises(FeedbackValidationError, match="disjoint"):
        validate_feedback_request(
            _request(
                review_action="corrected",
                accepted_labels=["Bug"],
                rejected_labels=["Bug"],
            ),
            make_test_bundle(),
        )


def test_validate_feedback_request_rejects_invalid_artifact_id_format() -> None:
    with pytest.raises(FeedbackValidationError, match="Invalid model_dataset_id format"):
        validate_feedback_request(
            _request(model_dataset_id="not-valid"),
            make_test_bundle(),
        )


def test_validate_feedback_request_rejects_artifact_id_mismatch() -> None:
    with pytest.raises(FeedbackValidationError, match="does not match loaded inference bundle"):
        validate_feedback_request(
            _request(),
            make_test_bundle(model_dataset_id=TEST_MODEL_DATASET_ID + "0"),
        )


def test_validate_feedback_request_rejects_accepted_action_with_mismatched_labels() -> None:
    with pytest.raises(FeedbackValidationError, match="accepted_labels to equal predicted_labels"):
        validate_feedback_request(
            _request(
                review_action="accepted",
                predicted_labels=["Indexing"],
                accepted_labels=["Bug"],
            ),
            make_test_bundle(),
        )


def test_validate_feedback_request_rejects_corrected_action_with_same_labels() -> None:
    with pytest.raises(FeedbackValidationError, match="accepted_labels to differ"):
        validate_feedback_request(
            _request(
                review_action="corrected",
                predicted_labels=["Indexing"],
                accepted_labels=["Indexing"],
            ),
            make_test_bundle(),
        )


def test_validate_feedback_request_rejects_rejected_action_with_accepted_labels() -> None:
    with pytest.raises(FeedbackValidationError, match="accepted_labels to be empty"):
        validate_feedback_request(
            _request(
                review_action="rejected",
                predicted_labels=["Indexing"],
                accepted_labels=["Bug"],
                rejected_labels=[],
            ),
            make_test_bundle(),
        )


def test_validate_feedback_request_accepts_rejected_action() -> None:
    validate_feedback_request(
        _request(
            review_action="rejected",
            predicted_labels=["Indexing"],
            accepted_labels=[],
            rejected_labels=["Indexing"],
        ),
        make_test_bundle(),
    )


def test_validate_feedback_request_accepts_accepted_action() -> None:
    validate_feedback_request(
        _request(
            review_action="accepted",
            predicted_labels=["Indexing"],
            accepted_labels=["Indexing"],
            rejected_labels=[],
        ),
        make_test_bundle(),
    )


def test_validate_feedback_request_rejects_rejected_action_wrong_rejected_labels() -> None:
    with pytest.raises(FeedbackValidationError, match="rejected_labels to equal predicted_labels"):
        validate_feedback_request(
            _request(
                review_action="rejected",
                predicted_labels=["Indexing"],
                accepted_labels=[],
                rejected_labels=["Bug"],
            ),
            make_test_bundle(),
        )
