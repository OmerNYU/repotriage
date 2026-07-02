"""Pydantic models, identity hashing, and domain exceptions for model-ready datasets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from repotriage.dataset.models import (
    DatasetId,
    Sha256Hex,
    _ensure_utc,
    format_utc_datetime,
)
from repotriage.label_policy.models import PolicyId
from repotriage.paths import resolve_within_directory

MODEL_DATASET_VERSION: Literal["1"] = "1"
MODEL_DATASET_MANIFEST_SCHEMA_VERSION: Literal["1"] = "1"
MODEL_READY_RECORD_SCHEMA_VERSION: Literal["1"] = "1"
LABEL_MAP_SCHEMA_VERSION: Literal["1"] = "1"
SPLIT_REPORT_SCHEMA_VERSION: Literal["1"] = "1"
TEMPORAL_SPLITTER_VERSION: Literal["1"] = "1"
TEXT_REPRESENTATION_VERSION: Literal["1"] = "1"

# Output-contract schema versions bound by MODEL_DATASET_VERSION. Any change to these
# schemas, support-validation semantics, or deterministic ordering requires bumping
# MODEL_DATASET_VERSION (and thus model_dataset_input_sha256 / model-dataset id).
OUTPUT_CONTRACT_VERSIONS = (
    MODEL_READY_RECORD_SCHEMA_VERSION,
    LABEL_MAP_SCHEMA_VERSION,
    SPLIT_REPORT_SCHEMA_VERSION,
)

MODEL_DATASET_OUTPUT_CONTRACTS: dict[str, dict[str, str]] = {
    MODEL_DATASET_VERSION: {
        "record_schema_version": MODEL_READY_RECORD_SCHEMA_VERSION,
        "label_map_schema_version": LABEL_MAP_SCHEMA_VERSION,
        "split_report_schema_version": SPLIT_REPORT_SCHEMA_VERSION,
        "text_representation_version": TEXT_REPRESENTATION_VERSION,
        "temporal_splitter_version": TEMPORAL_SPLITTER_VERSION,
    },
}

RECORDS_JSONL_FILE = "records.jsonl"
LABEL_MAP_JSON_FILE = "label_map.json"
SPLIT_REPORT_JSON_FILE = "split_report.json"
SPLIT_REPORT_MARKDOWN_FILE = "split_report.md"

MODEL_DATASET_ID_PATTERN = (
    r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-md[1-9]\d*-[0-9a-f]{12}$"
)
_MODEL_DATASET_ID_RE = re.compile(MODEL_DATASET_ID_PATTERN)

ModelDatasetId = Annotated[str, StringConstraints(pattern=MODEL_DATASET_ID_PATTERN)]

SplitName = Literal["train", "validation", "test"]

_FLOAT_ABS_TOL = 1e-12


def _floats_consistent(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)


def _require_strict_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a JSON integer")
    return value


def _require_strict_positive_int(value: object, *, field: str) -> int:
    parsed = _require_strict_int(value, field=field)
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive JSON integer")
    return parsed


def _require_strict_nonneg_int(value: object, *, field: str) -> int:
    parsed = _require_strict_int(value, field=field)
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative JSON integer")
    return parsed


def _require_strict_binary_int(value: object) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError("target_vector values must be JSON integer 0 or 1")
    return value


def _require_strict_int_dict(
    value: object, *, field: str, non_negative: bool = True
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    parser = _require_strict_nonneg_int if non_negative else _require_strict_int
    return {
        str(key): parser(item, field=f"{field} value")
        for key, item in value.items()
    }


def _is_safe_relative_path(value: str) -> bool:
    if not value:
        return False
    try:
        resolve_within_directory(Path("/__model_dataset_anchor__"), value)
    except ValueError:
        return False
    return True


class ModelDatasetError(RuntimeError):
    """Base class for model-dataset domain errors."""


class ModelDatasetConfigError(ModelDatasetError):
    """Raised when a temporal split configuration is invalid or incompatible."""


class ModelDatasetInputError(ModelDatasetError):
    """Raised when required dataset or policy inputs are missing or mismatched."""


class ModelDatasetTransformError(ModelDatasetError):
    """Raised when a normalized issue cannot be transformed into a model-ready record."""


class ModelDatasetSplitSupportError(ModelDatasetError):
    """Raised when split support validation fails a hard minimum."""


class ModelDatasetCorruptionError(ModelDatasetError):
    """Raised when an existing model-ready artifact is corrupt or incompatible."""


class ModelDatasetBuildError(ModelDatasetError):
    """Raised when staging or publication of a model-ready artifact fails."""


class ModelDatasetReadError(ModelDatasetError):
    """Raised when model-ready JSONL cannot be read or validated."""


def compute_model_dataset_input_sha256(
    *,
    model_dataset_version: str,
    dataset_id: str,
    dataset_output_sha256: str,
    policy_id: str,
    policy_json_sha256: str,
    text_representation_version: str,
    temporal_splitter_version: str,
    split_config_schema_version: str,
    split_config_sha256: str,
) -> str:
    """Derive the deterministic input hash binding all output-affecting inputs."""
    payload = {
        "dataset_id": dataset_id,
        "dataset_output_sha256": dataset_output_sha256,
        "model_dataset_version": model_dataset_version,
        "policy_id": policy_id,
        "policy_json_sha256": policy_json_sha256,
        "split_config_schema_version": split_config_schema_version,
        "split_config_sha256": split_config_sha256,
        "temporal_splitter_version": temporal_splitter_version,
        "text_representation_version": text_representation_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_model_dataset_id(
    dataset_id: str,
    model_dataset_input_sha256: str,
    model_dataset_version: str = MODEL_DATASET_VERSION,
) -> str:
    """Derive a content-aware model-dataset id from dataset id and input hash."""
    short_hash = model_dataset_input_sha256[:12]
    return f"{dataset_id}-md{model_dataset_version}-{short_hash}"


class ModelReadyRecord(BaseModel):
    """One model-ready issue record with feature text, targets, and split assignment."""

    schema_version: Literal["1"] = MODEL_READY_RECORD_SCHEMA_VERSION
    repository: str
    issue_id: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    created_at: datetime
    title: str
    body: str
    feature_text: str
    selected_labels: list[str] = Field(default_factory=list)
    target_vector: list[int] = Field(default_factory=list)
    split: SplitName

    @field_validator("issue_id", "issue_number", mode="before")
    @classmethod
    def strict_positive_ids(cls, value: object, info) -> int:
        return _require_strict_positive_int(value, field=info.field_name)

    @field_validator("target_vector", mode="before")
    @classmethod
    def strict_binary_vector(cls, value: object) -> list[int]:
        if not isinstance(value, list):
            raise ValueError("target_vector must be a list")
        return [_require_strict_binary_int(item) for item in value]

    @field_validator("created_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("created_at", when_used="json")
    def serialize_created_at(self, value: datetime) -> str:
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_record(self) -> ModelReadyRecord:
        if len(self.target_vector) != len(self.selected_labels):
            # selected_labels may be shorter than full target count; check binary values first
            pass
        for value in self.target_vector:
            if type(value) is not int or value not in (0, 1):
                raise ValueError("target_vector values must be JSON integer 0 or 1")
        return self


class LabelMap(BaseModel):
    """Canonical target-label ordering for a model-ready dataset."""

    schema_version: Literal["1"] = LABEL_MAP_SCHEMA_VERSION
    policy_id: PolicyId
    target_count: int = Field(ge=0)
    labels: list[str] = Field(default_factory=list)
    label_to_index: dict[str, int] = Field(default_factory=dict)

    @field_validator("target_count", mode="before")
    @classmethod
    def strict_target_count(cls, value: object) -> int:
        return _require_strict_nonneg_int(value, field="target_count")

    @field_validator("label_to_index", mode="before")
    @classmethod
    def strict_label_indices(cls, value: object) -> dict[str, int]:
        return _require_strict_int_dict(value, field="label_to_index", non_negative=True)

    @model_validator(mode="after")
    def validate_invariants(self) -> LabelMap:
        if len(self.labels) != self.target_count:
            raise ValueError("labels length must equal target_count")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must not contain duplicates")
        expected_map = {label: index for index, label in enumerate(self.labels)}
        if self.label_to_index != expected_map:
            raise ValueError("label_to_index must match labels order")
        return self


class SplitWarning(BaseModel):
    """Structured low-support warning for a label in a split."""

    model_config = ConfigDict(extra="forbid")

    code: Literal["low_positive_support"] = "low_positive_support"
    label: str
    split: SplitName
    count: int = Field(ge=0)
    threshold: int = Field(ge=1)

    @field_validator("count", "threshold", mode="before")
    @classmethod
    def strict_counts(cls, value: object, info) -> int:
        if info.field_name == "threshold":
            return _require_strict_positive_int(value, field=info.field_name)
        return _require_strict_nonneg_int(value, field=info.field_name)


class SplitStatistics(BaseModel):
    """Per-split issue and target statistics."""

    model_config = ConfigDict(extra="forbid")

    issue_count: int = Field(ge=0)
    fraction: float = Field(ge=0.0, le=1.0)
    earliest_created_at: datetime | None = None
    latest_created_at: datetime | None = None
    all_zero_target_count: int = Field(ge=0)
    target_cardinality_histogram: dict[str, int] = Field(default_factory=dict)
    positives_per_label: dict[str, int] = Field(default_factory=dict)

    @field_validator("issue_count", "all_zero_target_count", mode="before")
    @classmethod
    def strict_split_counts(cls, value: object, info) -> int:
        return _require_strict_nonneg_int(value, field=info.field_name)

    @field_validator("target_cardinality_histogram", "positives_per_label", mode="before")
    @classmethod
    def strict_count_maps(cls, value: object, info) -> dict[str, int]:
        return _require_strict_int_dict(value, field=info.field_name, non_negative=True)

    @field_validator("earliest_created_at", "latest_created_at")
    @classmethod
    def ensure_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _ensure_utc(value)

    @field_serializer("earliest_created_at", "latest_created_at", when_used="json")
    def serialize_datetimes(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_utc_datetime(value)


class GlobalTargetStatistics(BaseModel):
    """Dataset-wide target assignment statistics."""

    model_config = ConfigDict(extra="forbid")

    total_records: int = Field(ge=0)
    target_count: int = Field(ge=0)
    issues_with_included_target: int = Field(ge=0)
    issues_without_included_target: int = Field(ge=0)
    target_coverage_fraction: float = Field(ge=0.0, le=1.0)
    positive_assignments: int = Field(ge=0)
    all_zero_target_count: int = Field(ge=0)

    @field_validator(
        "total_records",
        "target_count",
        "issues_with_included_target",
        "issues_without_included_target",
        "positive_assignments",
        "all_zero_target_count",
        mode="before",
    )
    @classmethod
    def strict_global_counts(cls, value: object, info) -> int:
        return _require_strict_nonneg_int(value, field=info.field_name)


class SupportValidationSummary(BaseModel):
    """Summary of split-support validation outcome."""

    model_config = ConfigDict(extra="forbid")

    hard_errors: list[str] = Field(default_factory=list)
    warnings: list[SplitWarning] = Field(default_factory=list)


class SplitReport(BaseModel):
    """Deterministic split analysis report (no build timestamp)."""

    schema_version: Literal["1"] = SPLIT_REPORT_SCHEMA_VERSION
    split_strategy: str
    validation_start: datetime
    test_start: datetime
    boundary_semantics: dict[str, str]
    total_records: int = Field(ge=0)
    global_target_statistics: GlobalTargetStatistics
    splits: dict[str, SplitStatistics]
    warnings: list[SplitWarning] = Field(default_factory=list)
    support_validation: SupportValidationSummary

    @field_validator("total_records", mode="before")
    @classmethod
    def strict_total_records(cls, value: object) -> int:
        return _require_strict_nonneg_int(value, field="total_records")

    @field_validator("validation_start", "test_start")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("validation_start", "test_start", when_used="json")
    def serialize_cutoffs(self, value: datetime) -> str:
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_splits(self) -> SplitReport:
        required = {"train", "validation", "test"}
        if set(self.splits) != required:
            raise ValueError("splits must contain exactly train, validation, and test")
        total = sum(split.issue_count for split in self.splits.values())
        if total != self.total_records:
            raise ValueError("split issue counts must sum to total_records")
        return self


class ModelDatasetManifest(BaseModel):
    """Validated lineage manifest describing one immutable model-ready artifact."""

    schema_version: Literal["1"] = MODEL_DATASET_MANIFEST_SCHEMA_VERSION
    model_dataset_version: str
    model_dataset_id: ModelDatasetId
    model_dataset_input_sha256: Sha256Hex
    repository: str
    dataset_id: DatasetId
    dataset_output_sha256: Sha256Hex
    policy_id: PolicyId
    policy_json_sha256: Sha256Hex
    text_representation_version: str
    temporal_splitter_version: str
    split_config_schema_version: str
    split_config_sha256: Sha256Hex
    validation_start: datetime
    test_start: datetime
    built_at: datetime
    records_written: int = Field(ge=0)
    target_count: int = Field(ge=0)
    records_file: str = RECORDS_JSONL_FILE
    records_sha256: Sha256Hex
    label_map_file: str = LABEL_MAP_JSON_FILE
    label_map_sha256: Sha256Hex
    split_report_json_file: str = SPLIT_REPORT_JSON_FILE
    split_report_json_sha256: Sha256Hex
    split_report_markdown_file: str = SPLIT_REPORT_MARKDOWN_FILE
    split_report_markdown_sha256: Sha256Hex

    @field_validator("records_written", "target_count", mode="before")
    @classmethod
    def strict_manifest_counts(cls, value: object, info) -> int:
        return _require_strict_nonneg_int(value, field=info.field_name)

    @field_validator("built_at", "validation_start", "test_start")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("built_at", "validation_start", "test_start", when_used="json")
    def serialize_datetimes(self, value: datetime) -> str:
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> ModelDatasetManifest:
        expected_input = compute_model_dataset_input_sha256(
            model_dataset_version=self.model_dataset_version,
            dataset_id=self.dataset_id,
            dataset_output_sha256=self.dataset_output_sha256,
            policy_id=self.policy_id,
            policy_json_sha256=self.policy_json_sha256,
            text_representation_version=self.text_representation_version,
            temporal_splitter_version=self.temporal_splitter_version,
            split_config_schema_version=self.split_config_schema_version,
            split_config_sha256=self.split_config_sha256,
        )
        if self.model_dataset_input_sha256 != expected_input:
            raise ValueError(
                "model_dataset_input_sha256 is inconsistent with the input payload"
            )
        expected_id = compute_model_dataset_id(
            self.dataset_id,
            self.model_dataset_input_sha256,
            self.model_dataset_version,
        )
        if self.model_dataset_id != expected_id:
            raise ValueError(
                f"model_dataset_id {self.model_dataset_id!r} is inconsistent with "
                f"dataset_id and model_dataset_input_sha256 (expected {expected_id!r})"
            )
        for path_field in (
            "records_file",
            "label_map_file",
            "split_report_json_file",
            "split_report_markdown_file",
        ):
            if not _is_safe_relative_path(getattr(self, path_field)):
                raise ValueError(f"{path_field} must be a safe relative path")
        return self
