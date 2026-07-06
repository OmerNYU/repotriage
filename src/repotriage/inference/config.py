"""Human-authored inference configuration: schema, loading, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from repotriage.inference.models import InferenceConfigError

INFERENCE_CONFIG_SCHEMA_VERSION: Literal["1"] = "1"
INFERENCE_BUNDLE_VERSION: Literal["1"] = "1"


class InferenceConfigDocument(BaseModel):
    """Human-authored inference bundle binding configuration."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["1"] = INFERENCE_CONFIG_SCHEMA_VERSION
    inference_bundle_version: Literal["1"] = INFERENCE_BUNDLE_VERSION
    repository: str = Field(min_length=1)
    model_dataset_id: str = Field(min_length=1)
    text_representation_version: Literal["1"] = "1"
    baseline_run_id: str = Field(min_length=1)
    threshold_policy_id: str = Field(min_length=1)
    abstention_policy_id: str = Field(min_length=1)
    retrieval_run_id: str = Field(min_length=1)
    default_top_k: int = Field(ge=1)
    trust_serialized_models: bool = True


def load_inference_config(path: Path) -> InferenceConfigDocument:
    """Load and validate an inference configuration file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InferenceConfigError(f"Unable to read inference config at {path}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceConfigError(f"Invalid JSON in inference config at {path}: {exc}") from exc

    try:
        config = InferenceConfigDocument.model_validate(payload)
    except ValidationError as exc:
        raise InferenceConfigError(f"Invalid inference config at {path}: {exc}") from exc

    if config.inference_bundle_version != INFERENCE_BUNDLE_VERSION:
        raise InferenceConfigError(
            f"Unsupported inference_bundle_version {config.inference_bundle_version!r}; "
            f"expected {INFERENCE_BUNDLE_VERSION!r}."
        )

    if not config.trust_serialized_models:
        raise InferenceConfigError(
            "trust_serialized_models must be true to load local model and retrieval index files."
        )

    return config
