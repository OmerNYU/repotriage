"""Temporal split configuration: schema, loading, and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from repotriage.dataset.models import _ensure_utc, format_utc_datetime
from repotriage.model_dataset.models import ModelDatasetConfigError

CONFIG_SCHEMA_VERSION: Literal["1"] = "1"


class BoundarySemantics(BaseModel):
    """Documented interval semantics for each split."""

    model_config = ConfigDict(extra="forbid")

    train: str
    validation: str
    test: str


class MinimumPositiveSupport(BaseModel):
    """Minimum positive label counts required per split (hard errors when unmet)."""

    model_config = ConfigDict(extra="forbid")

    train: int = Field(ge=1)
    validation: int = Field(ge=1)
    test: int = Field(ge=1)


class TemporalSplitConfig(BaseModel):
    """Human-authored temporal split configuration for model-ready dataset builds."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["1"] = CONFIG_SCHEMA_VERSION
    repository: str = Field(min_length=1)
    split_strategy: Literal["temporal_calendar"] = "temporal_calendar"
    validation_start: datetime
    test_start: datetime
    boundary_semantics: BoundarySemantics
    minimum_positive_support: MinimumPositiveSupport
    low_support_warning_threshold: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> TemporalSplitConfig:
        validation_start = _ensure_utc(self.validation_start)
        test_start = _ensure_utc(self.test_start)
        if test_start <= validation_start:
            raise ValueError("test_start must be strictly after validation_start")
        return self

    @model_validator(mode="before")
    @classmethod
    def parse_datetimes(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        parsed = dict(data)
        for field in ("validation_start", "test_start"):
            value = parsed.get(field)
            if isinstance(value, str):
                parsed[field] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed


def canonical_config_bytes(config: TemporalSplitConfig) -> bytes:
    """Serialize the validated config to canonical UTF-8 JSON bytes for hashing."""
    payload = config.model_dump(mode="json")
    payload["validation_start"] = format_utc_datetime(_ensure_utc(config.validation_start))
    payload["test_start"] = format_utc_datetime(_ensure_utc(config.test_start))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def config_sha256(config: TemporalSplitConfig) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def load_split_config(config_path: Path) -> tuple[TemporalSplitConfig, str]:
    """Load and validate a temporal split configuration file."""
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelDatasetConfigError(
            f"Unable to read split configuration at {config_path}: {exc}"
        ) from exc
    try:
        config = TemporalSplitConfig.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ModelDatasetConfigError(
            f"Invalid split configuration at {config_path}: {exc}"
        ) from exc
    return config, config_sha256(config)
