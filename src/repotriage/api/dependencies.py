"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from repotriage.inference.artifact_loader import LoadedInferenceBundle
from repotriage.persistence.feedback_repository import FeedbackRepository


def get_bundle(request: Request) -> LoadedInferenceBundle:
    """Return the inference bundle loaded at application startup."""
    return request.app.state.bundle


def get_feedback_repository(request: Request) -> FeedbackRepository:
    """Return the feedback repository initialized at application startup."""
    return request.app.state.feedback_repository
