"""Pydantic models, version constants, and domain exceptions for dataset auditing.

The audit subsystem depends on the dataset contract (normalized issues and the
processed manifest) but never the other way around. Identity and lineage models bind
an audit artifact to the exact normalized dataset bytes it analyzed.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
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
from repotriage.paths import resolve_within_directory

AUDIT_VERSION: Literal["2"] = "2"
AUDIT_DOCUMENT_SCHEMA_VERSION: Literal["2"] = "2"
AUDIT_MANIFEST_SCHEMA_VERSION: Literal["2"] = "2"

AUDIT_JSON_FILE = "audit.json"
AUDIT_MARKDOWN_FILE = "audit.md"

# Absolute tolerance for reconciling derived floats (fractions, ratios) against their
# count-based definitions. Counts, ids, hashes, and versions are always compared exactly.
_FLOAT_ABS_TOL = 1e-12


def _floats_consistent(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)

# <dataset id>-a<version>, e.g. 20260628T161306010651Z-n1-074402d21505-a1
AUDIT_ID_PATTERN = r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-a[1-9]\d*$"

AuditId = Annotated[str, StringConstraints(pattern=AUDIT_ID_PATTERN)]

Severity = Literal["low", "medium", "high"]


class AuditError(RuntimeError):
    """Base class for audit-subsystem domain errors."""


class AuditDatasetError(AuditError):
    """Raised when the requested normalized dataset cannot be selected or read."""


class AuditReadError(AuditError):
    """Raised when streaming or validating the normalized JSONL fails."""


class AuditCorruptionError(AuditError):
    """Raised when an existing immutable audit artifact is corrupt or incompatible."""


def compute_audit_id(dataset_id: str, audit_version: str = AUDIT_VERSION) -> str:
    """Derive the deterministic, immutable audit id from a dataset id and version."""
    return f"{dataset_id}-a{audit_version}"


def _is_safe_relative_path(value: str) -> bool:
    if not value:
        return False
    try:
        resolve_within_directory(Path("/__audit_anchor__"), value)
    except ValueError:
        return False
    return True


class CountFraction(BaseModel):
    """A non-negative count paired with its fraction of the issue population."""

    count: int = Field(ge=0)
    fraction: float = Field(ge=0.0, le=1.0)


class TextFieldStats(BaseModel):
    """Distribution summary for a character-length population.

    All fields are ``None`` only when the population is empty.
    """

    min: int | None = None
    median: float | None = None
    mean: float | None = None
    p90: float | None = None
    p95: float | None = None
    max: int | None = None


class LabelsPerIssueStats(BaseModel):
    """Distribution summary for the number of labels per issue."""

    min: int = Field(ge=0)
    median: float = Field(ge=0.0)
    mean: float = Field(ge=0.0)
    max: int = Field(ge=0)


class RareLabelBuckets(BaseModel):
    """Counts of labels whose support (issue frequency) is below each threshold."""

    lt_5: int = Field(ge=0)
    lt_10: int = Field(ge=0)
    lt_20: int = Field(ge=0)
    lt_50: int = Field(ge=0)
    lt_100: int = Field(ge=0)


class LabelFrequency(BaseModel):
    """Frequency and temporal span of a single label across the dataset."""

    name: str
    count: int = Field(ge=1)
    fraction: float = Field(ge=0.0, le=1.0)
    first_created_at: datetime
    last_created_at: datetime

    @field_validator("first_created_at", "last_created_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("first_created_at", "last_created_at", when_used="json")
    def serialize_datetimes(self, value: datetime) -> str:
        return format_utc_datetime(value)


class LabelPair(BaseModel):
    """An unordered label pair and its co-occurrence count."""

    label_a: str
    label_b: str
    count: int = Field(ge=2)

    @model_validator(mode="after")
    def canonical_order(self) -> LabelPair:
        if not self.label_a < self.label_b:
            raise ValueError("label_a must sort strictly before label_b")
        return self


class TextStructuralIndicators(BaseModel):
    """Issue-level structural indicators counted over the whole population."""

    empty_bodies: CountFraction
    short_bodies_lt_100: CountFraction
    long_bodies_gt_10000: CountFraction
    with_code_fence: CountFraction
    with_url: CountFraction
    with_heading: CountFraction


class DatasetIdentity(BaseModel):
    """Immutable identity and lineage binding an audit to its normalized dataset."""

    audit_version: str
    audit_id: AuditId
    repository: str
    dataset_id: DatasetId
    dataset_output_sha256: Sha256Hex
    issue_schema_version: str
    normalizer_version: str

    @model_validator(mode="after")
    def audit_id_consistent(self) -> DatasetIdentity:
        expected = compute_audit_id(self.dataset_id, self.audit_version)
        if self.audit_id != expected:
            raise ValueError(
                f"audit_id {self.audit_id!r} is inconsistent with dataset_id and "
                f"audit_version (expected {expected!r})"
            )
        return self


class RepositorySummary(BaseModel):
    """Objective repository-level metrics for the audited dataset."""

    total_issues: int = Field(ge=0)
    labelled_issues: int = Field(ge=0)
    unlabelled_issues: int = Field(ge=0)
    labelled_fraction: float = Field(ge=0.0, le=1.0)
    unlabelled_fraction: float = Field(ge=0.0, le=1.0)
    open_issues: int = Field(ge=0)
    closed_issues: int = Field(ge=0)
    null_author_issues: int = Field(ge=0)
    earliest_created_at: datetime | None = None
    latest_created_at: datetime | None = None
    temporal_span_days: float = Field(ge=0.0)
    active_month_count: int = Field(ge=0)
    calendar_span_months: int = Field(ge=0)

    @field_validator("earliest_created_at", "latest_created_at")
    @classmethod
    def ensure_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _ensure_utc(value)

    @field_serializer("earliest_created_at", "latest_created_at", when_used="json")
    def serialize_datetimes(self, value: datetime | None) -> str | None:
        return None if value is None else format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> RepositorySummary:
        if self.labelled_issues + self.unlabelled_issues != self.total_issues:
            raise ValueError("labelled_issues + unlabelled_issues must equal total_issues")
        if self.open_issues + self.closed_issues != self.total_issues:
            raise ValueError("open_issues + closed_issues must equal total_issues")
        if self.null_author_issues > self.total_issues:
            raise ValueError("null_author_issues must not exceed total_issues")
        if (self.earliest_created_at is None) != (self.latest_created_at is None):
            raise ValueError("earliest_created_at and latest_created_at must both be set or unset")
        if (
            self.earliest_created_at is not None
            and self.latest_created_at is not None
            and self.latest_created_at < self.earliest_created_at
        ):
            raise ValueError("latest_created_at must not precede earliest_created_at")
        if self.active_month_count > self.calendar_span_months:
            raise ValueError("active_month_count must not exceed calendar_span_months")
        if self.total_issues > 0:
            expected_labelled = self.labelled_issues / self.total_issues
            if not _floats_consistent(self.labelled_fraction, expected_labelled):
                raise ValueError(
                    "labelled_fraction is inconsistent with labelled_issues/total_issues"
                )
            expected_unlabelled = self.unlabelled_issues / self.total_issues
            if not _floats_consistent(self.unlabelled_fraction, expected_unlabelled):
                raise ValueError(
                    "unlabelled_fraction is inconsistent with unlabelled_issues/total_issues"
                )
        return self


class TextMetrics(BaseModel):
    """Character-length distributions and structural indicators."""

    title_chars: TextFieldStats
    body_chars: TextFieldStats
    total_text_chars: TextFieldStats
    structural: TextStructuralIndicators


class LabelMetrics(BaseModel):
    """Objective label-distribution metrics."""

    unique_label_count: int = Field(ge=0)
    total_label_assignments: int = Field(ge=0)
    zero_label_issue_count: int = Field(ge=0)
    labels_per_issue: LabelsPerIssueStats
    label_cardinality: float = Field(ge=0.0)
    label_density: float = Field(ge=0.0)
    rare_label_buckets: RareLabelBuckets
    labels: list[LabelFrequency] = Field(default_factory=list)
    label_pairs: list[LabelPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_invariants(self) -> LabelMetrics:
        if len(self.labels) != self.unique_label_count:
            raise ValueError("labels length must equal unique_label_count")
        if self.unique_label_count > 0 and not _floats_consistent(
            self.label_density, self.label_cardinality / self.unique_label_count
        ):
            raise ValueError(
                "label_density is inconsistent with label_cardinality/unique_label_count"
            )
        return self


class TemporalMetrics(BaseModel):
    """Monthly temporal coverage in UTC."""

    earliest_created_at: datetime | None = None
    latest_created_at: datetime | None = None
    active_month_count: int = Field(ge=0)
    calendar_span_months: int = Field(ge=0)
    issues_by_month: dict[str, int] = Field(default_factory=dict)
    labelled_issues_by_month: dict[str, int] = Field(default_factory=dict)

    @field_validator("earliest_created_at", "latest_created_at")
    @classmethod
    def ensure_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _ensure_utc(value)

    @field_serializer("earliest_created_at", "latest_created_at", when_used="json")
    def serialize_datetimes(self, value: datetime | None) -> str | None:
        return None if value is None else format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> TemporalMetrics:
        if len(self.issues_by_month) != self.active_month_count:
            raise ValueError("issues_by_month size must equal active_month_count")
        if self.active_month_count > self.calendar_span_months:
            raise ValueError("active_month_count must not exceed calendar_span_months")
        if set(self.labelled_issues_by_month) != set(self.issues_by_month):
            raise ValueError("labelled_issues_by_month must share keys with issues_by_month")
        return self


class SuitabilityWarning(BaseModel):
    """A heuristic, versioned suitability warning. Never an aggregate quality score."""

    code: str
    severity: Severity
    value: float
    threshold: float
    message: str


class AuditDocument(BaseModel):
    """The full, deterministic, machine-readable audit (the ``audit.json`` content).

    This document intentionally contains no build timestamp so that its bytes depend
    only on the normalized dataset and the audit version.
    """

    schema_version: Literal["2"] = AUDIT_DOCUMENT_SCHEMA_VERSION
    identity: DatasetIdentity
    repository_summary: RepositorySummary
    text_metrics: TextMetrics
    label_metrics: LabelMetrics
    temporal_metrics: TemporalMetrics
    warnings: list[SuitabilityWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cross_section(self) -> AuditDocument:
        total = self.repository_summary.total_issues
        if total <= 0:
            return self

        labels = self.label_metrics
        if not _floats_consistent(labels.label_cardinality, labels.total_label_assignments / total):
            raise ValueError(
                "label_cardinality is inconsistent with total_label_assignments/total_issues"
            )

        structural = self.text_metrics.structural
        for name in (
            "empty_bodies",
            "short_bodies_lt_100",
            "long_bodies_gt_10000",
            "with_code_fence",
            "with_url",
            "with_heading",
        ):
            indicator = getattr(structural, name)
            if not _floats_consistent(indicator.fraction, indicator.count / total):
                raise ValueError(f"{name} fraction is inconsistent with count/total_issues")

        for label in labels.labels:
            if not _floats_consistent(label.fraction, label.count / total):
                raise ValueError(
                    f"label {label.name!r} fraction is inconsistent with count/total_issues"
                )
        return self


class AuditManifest(BaseModel):
    """Validated lineage manifest describing one immutable audit artifact."""

    schema_version: Literal["2"] = AUDIT_MANIFEST_SCHEMA_VERSION
    audit_document_schema_version: Literal["2"] = AUDIT_DOCUMENT_SCHEMA_VERSION
    audit_version: str
    audit_id: AuditId
    repository: str
    dataset_id: DatasetId
    dataset_output_sha256: Sha256Hex
    issue_schema_version: str
    normalizer_version: str
    built_at: datetime
    issues_analyzed: int = Field(ge=0)
    audit_json_file: str = AUDIT_JSON_FILE
    audit_json_sha256: Sha256Hex
    audit_markdown_file: str = AUDIT_MARKDOWN_FILE
    audit_markdown_sha256: Sha256Hex

    @field_validator("built_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("built_at", when_used="json")
    def serialize_built_at(self, value: datetime) -> str:
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> AuditManifest:
        expected = compute_audit_id(self.dataset_id, self.audit_version)
        if self.audit_id != expected:
            raise ValueError(
                f"audit_id {self.audit_id!r} is inconsistent with dataset_id and "
                f"audit_version (expected {expected!r})"
            )
        if not _is_safe_relative_path(self.audit_json_file):
            raise ValueError(
                f"audit_json_file must be a safe relative path: {self.audit_json_file!r}"
            )
        if not _is_safe_relative_path(self.audit_markdown_file):
            raise ValueError(
                f"audit_markdown_file must be a safe relative path: {self.audit_markdown_file!r}"
            )
        return self

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)
