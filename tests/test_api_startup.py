"""Tests for API startup failure behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repotriage.api.app import create_app
from repotriage.api.settings import ApiSettings
from repotriage.inference.models import InferenceConfigError


def test_startup_fails_when_inference_config_missing() -> None:
    settings = ApiSettings(inference_config_path=Path("nonexistent-config.json"))
    app = create_app(settings=settings)

    with pytest.raises(InferenceConfigError, match="Unable to read inference config"):
        with TestClient(app):
            pass
