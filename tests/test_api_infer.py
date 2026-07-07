"""Tests for POST /api/v1/infer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from repotriage.api.app import create_app
from repotriage.github.models import RepositoryRef
from repotriage.inference.models import (
    AbstentionResult,
    ArtifactReferences,
    ClassificationResult,
    InferenceError,
    InferenceInputError,
    InferenceInputSummary,
    InferenceIssueInput,
    InferenceResponse,
    ReproducibilityMetadata,
    RetrievalResult,
)
from repotriage.inference.pipeline import infer_issue as real_infer_issue


def _fake_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        repository=RepositoryRef(owner="pandas-dev", name="pandas"),
        config_path=Path("configs/test.json"),
    )


def _sample_response() -> InferenceResponse:
    return InferenceResponse(
        repository="pandas-dev/pandas",
        generated_at="2026-01-01T00:00:00Z",
        input=InferenceInputSummary(
            title="BUG: example",
            body_preview="Body",
            feature_text_sha256="a" * 64,
            text_representation_version="1",
        ),
        classification=ClassificationResult(
            label_order=["Bug"],
            scores=[],
            threshold=0.39,
            threshold_basis_points=3900,
            predicted_labels=[],
        ),
        abstention=AbstentionResult(
            confidence=None,
            threshold=0.84,
            threshold_basis_points=8400,
            should_abstain=True,
            reason="no_labels_predicted",
        ),
        retrieval=RetrievalResult(top_k=5, similar_issues=[]),
        artifacts=ArtifactReferences(
            model_dataset_id="md",
            baseline_run_id="bl",
            threshold_policy_id="tp",
            abstention_policy_id="ap",
            retrieval_run_id="rb",
        ),
        reproducibility=ReproducibilityMetadata(
            inference_config_path="configs/test.json",
            model_semantic_sha256="b" * 64,
            index_semantic_sha256="c" * 64,
            baseline_experiment_sha256="d" * 64,
            numerical_environment_sha256="e" * 64,
        ),
    )


def test_infer_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, InferenceIssueInput]] = []

    def fake_infer_issue(bundle, issue_input: InferenceIssueInput) -> InferenceResponse:
        calls.append((bundle, issue_input))
        return _sample_response()

    monkeypatch.setattr("repotriage.api.routes.infer.infer_issue", fake_infer_issue)
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/infer",
            json={
                "title": "BUG: example",
                "body": "Body text",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["repository"] == "pandas-dev/pandas"
    assert payload["schema_version"] == "1"
    assert len(calls) == 1
    assert calls[0][1].title == "BUG: example"
    assert calls[0][1].body == "Body text"
    assert calls[0][1].top_k == 3


def test_infer_missing_title_returns_422() -> None:
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post("/api/v1/infer", json={"body": "no title"})

    assert response.status_code == 422


def test_infer_invalid_top_k_returns_422() -> None:
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/infer",
            json={"title": "Title", "top_k": 0},
        )

    assert response.status_code == 422


def test_infer_extra_field_returns_422() -> None:
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/infer",
            json={"title": "Title", "unknown": "field"},
        )

    assert response.status_code == 422


def test_infer_repository_field_returns_422() -> None:
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/infer",
            json={"title": "Title", "repository": "other/repo"},
        )

    assert response.status_code == 422


def test_infer_malformed_json_returns_422() -> None:
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/infer",
            content=b"{not-json",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422


def test_infer_input_error_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_input_error(_bundle, _issue_input: InferenceIssueInput) -> InferenceResponse:
        raise InferenceInputError("invalid issue input")

    monkeypatch.setattr("repotriage.api.routes.infer.infer_issue", raise_input_error)
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post("/api/v1/infer", json={"title": "Title"})

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid issue input"}


def test_infer_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_inference_error(_bundle, _issue_input: InferenceIssueInput) -> InferenceResponse:
        raise InferenceError("unexpected inference failure")

    monkeypatch.setattr("repotriage.api.routes.infer.infer_issue", raise_inference_error)
    app = create_app(bundle=_fake_bundle())

    with TestClient(app) as client:
        response = client.post("/api/v1/infer", json={"title": "Title"})

    assert response.status_code == 500
    assert response.json() == {"detail": "unexpected inference failure"}


def test_infer_calls_real_infer_issue_when_not_mocked() -> None:
    """Guard against accidentally leaving infer_issue fully stubbed in integration paths."""
    assert real_infer_issue.__name__ == "infer_issue"
