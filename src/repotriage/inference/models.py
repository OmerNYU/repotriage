"""Pydantic models, reason enums, and domain exceptions for local issue inference."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from repotriage.abstention_policy.models import CONFIDENCE_DEFINITION_MAX_PREDICTED

INFERENCE_RESPONSE_SCHEMA_VERSION: Literal["1"] = "1"

AbstentionReason = Literal[
    "no_labels_predicted",
    "confidence_below_threshold",
    "confidence_meets_threshold",
]

InferenceWarning = Literal[
    "empty_title",
    "empty_body",
    "no_labels_predicted",
]


class InferenceError(RuntimeError):
    """Base class for inference domain errors."""


class InferenceConfigError(InferenceError):
    """Raised when the human-authored inference configuration is invalid."""


class InferenceInputError(InferenceError):
    """Raised when inference input is invalid."""


class InferenceBundleError(InferenceError):
    """Raised when artifact loading or compatibility checks fail."""


class InferenceIssueInput(BaseModel):
    """Input for scoring one new issue-like record."""

    model_config = ConfigDict(extra="forbid")

    title: str
    body: str = ""
    top_k: int | None = Field(default=None, ge=1)
    issue_number: int | None = Field(default=None, gt=0)
    issue_id: int | None = Field(default=None, gt=0)


class LabelScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    score: float = Field(ge=0.0, le=1.0)


class PredictedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    score: float = Field(ge=0.0, le=1.0)


class InferenceInputSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    body_preview: str
    feature_text_sha256: str = Field(min_length=64, max_length=64)
    text_representation_version: str


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_order: list[str]
    scores: list[LabelScore]
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_basis_points: int = Field(ge=0)
    predicted_labels: list[PredictedLabel]


class AbstentionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_method: Literal["max_predicted_label_score"] = CONFIDENCE_DEFINITION_MAX_PREDICTED
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_basis_points: int = Field(ge=0)
    should_abstain: bool
    reason: AbstentionReason


class SimilarIssueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    issue_id: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    similarity: float = Field(ge=-1.0, le=1.0)
    neighbor_selected_labels: list[str] = Field(default_factory=list)
    predicted_label_overlap: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["tfidf_cosine"] = "tfidf_cosine"
    top_k: int = Field(ge=1)
    similar_issues: list[SimilarIssueResult] = Field(default_factory=list)


class ArtifactReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_dataset_id: str
    baseline_run_id: str
    threshold_policy_id: str
    abstention_policy_id: str
    retrieval_run_id: str


class ReproducibilityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inference_config_path: str
    model_semantic_sha256: str
    index_semantic_sha256: str
    baseline_experiment_sha256: str
    numerical_environment_sha256: str
    serialization_security_warning: str | None = None


class InferenceResponse(BaseModel):
    """Combined intelligence response for one new issue."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = INFERENCE_RESPONSE_SCHEMA_VERSION
    repository: str
    generated_at: str
    input: InferenceInputSummary
    classification: ClassificationResult
    abstention: AbstentionResult
    retrieval: RetrievalResult
    artifacts: ArtifactReferences
    reproducibility: ReproducibilityMetadata
    warnings: list[InferenceWarning] = Field(default_factory=list)
