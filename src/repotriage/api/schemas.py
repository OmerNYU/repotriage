"""API-specific request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from repotriage.inference.models import InferenceIssueInput
from repotriage.persistence.schemas import FeedbackResponse

InferRequest = InferenceIssueInput

__all__ = ["FeedbackResponse", "HealthResponse", "InferRequest"]


class HealthResponse(BaseModel):
    """Liveness response including loaded bundle metadata."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    schema_version: Literal["1"] = "1"
    repository: str
    inference_config_path: str
