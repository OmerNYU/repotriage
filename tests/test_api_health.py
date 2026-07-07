"""Tests for GET /health."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from repotriage.api.app import create_app
from repotriage.github.models import RepositoryRef
from tests.helpers import noop_feedback_repository


def _fake_bundle(*, config_path: str = "configs/test.json"):
    return SimpleNamespace(
        repository=RepositoryRef(owner="pandas-dev", name="pandas"),
        config_path=Path(config_path),
    )


def test_health_returns_ok_with_bundle_metadata() -> None:
    bundle = _fake_bundle()
    app = create_app(bundle=bundle, feedback_repository=noop_feedback_repository())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "schema_version": "1",
        "repository": "pandas-dev/pandas",
        "inference_config_path": "configs/test.json",
    }
