"""Pydantic schemas for maintainer feedback requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FEEDBACK_SCHEMA_VERSION: Literal["1"] = "1"
MAX_REVIEWER_NOTE_LENGTH = 4000

ReviewAction = Literal["accepted", "corrected", "rejected"]


class InferenceArtifactsInput(BaseModel):
    """Inference artifact IDs that produced the prediction under review."""

    model_config = ConfigDict(extra="forbid")

    model_dataset_id: str
    baseline_run_id: str
    threshold_policy_id: str
    abstention_policy_id: str
    retrieval_run_id: str


class FeedbackRequest(BaseModel):
    """Maintainer review event submitted after inspecting a prediction."""

    model_config = ConfigDict(extra="forbid")

    feedback_schema_version: Literal["1"] = FEEDBACK_SCHEMA_VERSION
    repository: str
    issue_number: int = Field(gt=0)
    issue_title: str = Field(min_length=1)
    issue_body_preview: str = Field(default="", max_length=200)
    predicted_labels: list[str]
    accepted_labels: list[str]
    rejected_labels: list[str] = Field(default_factory=list)
    review_action: ReviewAction
    reviewer_note: str | None = Field(default=None, max_length=MAX_REVIEWER_NOTE_LENGTH)
    inference_artifacts: InferenceArtifactsInput


class FeedbackResponse(BaseModel):
    """Acknowledgement that a feedback event was stored."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str
    created_at: str
    status: Literal["stored"] = "stored"
