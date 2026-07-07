"""Tests for POST /api/v1/feedback."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from repotriage.api.app import create_app
from repotriage.persistence.errors import PersistenceError
from repotriage.persistence.schemas import FeedbackRequest, FeedbackResponse
from tests.helpers import (
    TEST_MODEL_DATASET_ID,
    make_feedback_request_payload,
    make_test_bundle,
    noop_feedback_repository,
)


def _recording_repository() -> SimpleNamespace:
    calls: list[FeedbackRequest] = []

    def store(body: FeedbackRequest) -> FeedbackResponse:
        calls.append(body)
        return FeedbackResponse(
            feedback_id="11111111-1111-1111-1111-111111111111",
            created_at="2026-01-01T00:00:00Z",
        )

    repo = SimpleNamespace(store=store, dispose=lambda: None)
    repo.calls = calls
    return repo


def test_feedback_success_returns_201() -> None:
    repository = _recording_repository()
    app = create_app(bundle=make_test_bundle(), feedback_repository=repository)

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=make_feedback_request_payload())

    assert response.status_code == 201
    payload = response.json()
    assert payload == {
        "feedback_id": "11111111-1111-1111-1111-111111111111",
        "created_at": "2026-01-01T00:00:00Z",
        "status": "stored",
    }
    assert len(repository.calls) == 1


def test_feedback_missing_issue_number_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    del payload["issue_number"]

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422


def test_feedback_empty_issue_title_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["issue_title"] = ""

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422


def test_feedback_extra_field_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["unknown"] = "field"

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422


def test_feedback_repository_mismatch_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload(repository="other/repo")

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422
    assert "does not match server-bound" in response.json()["detail"]


def test_feedback_unknown_label_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload(predicted_labels=["NotARealLabel"])

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422
    assert "Unknown label" in response.json()["detail"]


def test_feedback_invalid_review_action_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["review_action"] = "maybe"

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422


def test_feedback_persistence_error_returns_500() -> None:
    def raise_persistence_error(_body: FeedbackRequest) -> FeedbackResponse:
        raise PersistenceError("database write failed")

    repository = SimpleNamespace(store=raise_persistence_error, dispose=lambda: None)
    app = create_app(bundle=make_test_bundle(), feedback_repository=repository)

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=make_feedback_request_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to store feedback."}


def test_feedback_malformed_json_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback",
            content=b"{not-json",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422


def test_feedback_artifact_mismatch_returns_422() -> None:
    app = create_app(
        bundle=make_test_bundle(model_dataset_id=TEST_MODEL_DATASET_ID + "0"),
        feedback_repository=noop_feedback_repository(),
    )
    payload = make_feedback_request_payload()

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422
    assert "does not match loaded inference bundle" in response.json()["detail"]


def test_feedback_invalid_artifact_id_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["inference_artifacts"]["model_dataset_id"] = "not-valid"

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422
    assert "Invalid model_dataset_id format" in response.json()["detail"]


def test_feedback_issue_number_zero_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["issue_number"] = 0

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422


def test_feedback_body_preview_too_long_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["issue_body_preview"] = "x" * 201

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422


def test_feedback_reviewer_note_too_long_returns_422() -> None:
    app = create_app(bundle=make_test_bundle(), feedback_repository=noop_feedback_repository())
    payload = make_feedback_request_payload()
    payload["reviewer_note"] = "x" * 4001

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=payload)

    assert response.status_code == 422
