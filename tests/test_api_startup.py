"""Tests for API startup failure behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repotriage.api.app import create_app
from repotriage.api.settings import ApiSettings
from repotriage.inference.models import InferenceConfigError
from repotriage.persistence.errors import DatabaseUnavailableError
from tests.helpers import make_test_bundle, noop_feedback_repository


def test_startup_fails_when_inference_config_missing() -> None:
    settings = ApiSettings(inference_config_path=Path("nonexistent-config.json"))
    app = create_app(settings=settings, feedback_repository=noop_feedback_repository())

    with pytest.raises(InferenceConfigError, match="Unable to read inference config"):
        with TestClient(app):
            pass


def test_startup_fails_when_database_url_invalid(tmp_path: Path) -> None:
    settings = ApiSettings(
        inference_config_path=Path("configs/test.json"),
        database_url=f"sqlite:///{tmp_path}",
    )
    app = create_app(settings=settings, bundle=make_test_bundle())

    with pytest.raises(DatabaseUnavailableError, match="Database"):
        with TestClient(app):
            pass
