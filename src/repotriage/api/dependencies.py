"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from repotriage.inference.artifact_loader import LoadedInferenceBundle


def get_bundle(request: Request) -> LoadedInferenceBundle:
    """Return the inference bundle loaded at application startup."""
    return request.app.state.bundle
