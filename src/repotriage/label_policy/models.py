"""Pydantic models, version constants, controlled enums, and domain exceptions for the
target-label policy subsystem (lp2 contract).

Policy identity is content-addressed over every output-affecting input: the normalized
dataset bytes, the exact audit artifact bytes, the audit id, the configuration schema, and
the canonical configuration hash. These are folded into a deterministic
``policy_input_sha256`` whose 12-hex prefix forms the policy id, so changing any input
(dataset bytes, audit json bytes, audit id, configuration semantics, configuration schema,
or policy version) changes policy identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from repotriage.audit.models import AuditId
from repotriage.dataset.models import (
    DatasetId,
    Sha256Hex,
    _ensure_utc,
    format_utc_datetime,
)
from repotriage.paths import resolve_within_directory

LABEL_POLICY_VERSION: Literal["2"] = "2"
LABEL_POLICY_DOCUMENT_SCHEMA_VERSION: Literal["2"] = "2"
LABEL_POLICY_MANIFEST_SCHEMA_VERSION: Literal["2"] = "2"

LABEL_POLICY_JSON_FILE = "label_policy.json"
LABEL_POLICY_MARKDOWN_FILE = "label_policy.md"

# Audit artifacts this policy version is allowed to consume. Membership is explicit so a
# newly minted audit contract (for example a3) is never auto-accepted merely because the
# audit package's current constant changed.
SUPPORTED_AUDIT_VERSIONS: frozenset[str] = frozenset({"2"})
SUPPORTED_AUDIT_DOCUMENT_SCHEMA_VERSIONS: frozenset[str] = frozenset({"2"})

# Configuration schema this policy version accepts.
SUPPORTED_CONFIG_SCHEMA_VERSIONS: frozenset[str] = frozenset({"2"})

# Absolute tolerance for reconciling derived floats (fractions, ratios) against their
# count-based definitions. Counts, ids, hashes, and versions are always compared exactly.
_FLOAT_ABS_TOL = 1e-12

# <dataset id>-lp<version>-<12 hex policy-input-hash prefix>,
# e.g. 20260628T161306010651Z-n1-074402d21505-lp2-ab12cd34ef56
POLICY_ID_PATTERN = r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-lp[1-9]\d*-[0-9a-f]{12}$"

PolicyId = Annotated[str, StringConstraints(pattern=POLICY_ID_PATTERN)]

# Controlled vocabularies. Free-form explanations exist in addition to these, never
# instead of them. Semantic role and reason are never inferred from a label's name.
Decision = Literal["include", "defer", "exclude"]
DecisionSource = Literal["explicit", "default"]
LeakageRisk = Literal["low", "medium", "high"]
Role = Literal[
    "issue_type",
    "component",
    "quality",
    "infrastructure",
    "diagnostic",
    "workflow",
    "resolution",
    "design",
    "lifecycle",
    "unreviewed",
    "other",
]
ReasonCode = Literal[
    "selected_target",
    "insufficient_total_support",
    "insufficient_active_months",
    "insufficient_recent_support",
    "workflow_label",
    "post_investigation_outcome",
    "manual_deferral",
    "unreviewed_default",
]

_DEFERRAL_REASONS: frozenset[str] = frozenset(
    {
        "insufficient_total_support",
        "insufficient_active_months",
        "insufficient_recent_support",
        "manual_deferral",
    }
)
_EXPLICIT_EXCLUSION_REASONS: frozenset[str] = frozenset(
    {"workflow_label", "post_investigation_outcome"}
)

# Month keys are "%Y-%m"; a label that never occurs has no first/last month.
MonthKey = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}$")]


def _floats_consistent(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)


class LabelPolicyError(RuntimeError):
    """Base class for label-policy domain errors."""


class LabelPolicyConfigError(LabelPolicyError):
    """Raised when the human-authored configuration is missing, invalid, or inconsistent."""


class LabelPolicyInputError(LabelPolicyError):
    """Raised when the dataset/audit inputs are missing, mismatched, or disagree."""


class LabelPolicyCorruptionError(LabelPolicyError):
    """Raised when an existing immutable policy artifact is corrupt or incompatible."""


def validate_decision_reason(
    *,
    decision: str,
    reason_code: str,
    decision_source: str,
    explanation: str,
    role: str,
) -> None:
    """Enforce that decision, reason code, source, and role form a coherent combination.

    Raises ``ValueError`` (so it surfaces as a Pydantic validation error inside models and
    can be wrapped as a config error at load time).
    """
    if decision_source == "default":
        if decision != "exclude" or role != "unreviewed" or reason_code != "unreviewed_default":
            raise ValueError(
                "default-sourced decisions must be exclude/unreviewed/unreviewed_default"
            )
        return

    # Explicit (human-reviewed) decisions.
    if reason_code == "unreviewed_default":
        raise ValueError("unreviewed_default is reserved for default-sourced decisions")
    if decision == "include":
        if reason_code != "selected_target":
            raise ValueError("include decisions require reason_code 'selected_target'")
    elif decision == "defer":
        if reason_code not in _DEFERRAL_REASONS:
            raise ValueError(
                "defer decisions require an insufficient_* reason or manual_deferral"
            )
        if reason_code == "manual_deferral" and not explanation.strip():
            raise ValueError("manual_deferral requires a non-blank explanation")
    elif decision == "exclude":
        if reason_code not in _EXPLICIT_EXCLUSION_REASONS:
            raise ValueError(
                "explicit exclude decisions require 'workflow_label' or "
                "'post_investigation_outcome'"
            )


def compute_policy_input_sha256(
    *,
    policy_version: str,
    dataset_id: str,
    dataset_output_sha256: str,
    audit_id: str,
    audit_json_sha256: str,
    config_schema_version: str,
    config_sha256: str,
) -> str:
    """Hash the canonical policy-input payload that fully determines the generated policy."""
    payload = {
        "policy_version": policy_version,
        "dataset_id": dataset_id,
        "dataset_output_sha256": dataset_output_sha256,
        "audit_id": audit_id,
        "audit_json_sha256": audit_json_sha256,
        "config_schema_version": config_schema_version,
        "config_sha256": config_sha256,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_policy_id(
    dataset_id: str, policy_input_sha256: str, policy_version: str = LABEL_POLICY_VERSION
) -> str:
    """Derive the deterministic, immutable policy id from the policy-input hash."""
    return f"{dataset_id}-lp{policy_version}-{policy_input_sha256[:12]}"


def _is_safe_relative_path(value: str) -> bool:
    if not value:
        return False
    try:
        resolve_within_directory(Path("/__label_policy_anchor__"), value)
    except ValueError:
        return False
    return True


class SelectionCriteria(BaseModel):
    """Objective thresholds an included label must satisfy (inclusive boundaries).

    Defined here (rather than in ``config``) so the generated document can embed the exact,
    validated criteria without a circular import. ``config`` re-uses this model.
    """

    model_config = ConfigDict(extra="forbid")

    min_total_support: int = Field(gt=0)
    min_active_months: int = Field(gt=0)
    min_recent_support: int = Field(gt=0)
    recent_window_months: int = Field(gt=0)


class LabelsPerIssueStats(BaseModel):
    """Distribution summary for a per-issue count population (zeros included)."""

    min: int = Field(ge=0)
    median: float = Field(ge=0.0)
    mean: float = Field(ge=0.0)
    max: int = Field(ge=0)


class LabelDecisionRecord(BaseModel):
    """One resolved decision for a single audited label, enriched with objective facts."""

    label: str
    decision: Decision
    decision_source: DecisionSource
    role: Role
    leakage_risk: LeakageRisk
    reason_code: ReasonCode
    explanation: str
    total_support: int = Field(ge=0)
    issue_fraction: float = Field(ge=0.0, le=1.0)
    active_month_count: int = Field(ge=0)
    first_month: MonthKey | None = None
    last_month: MonthKey | None = None
    recent_support: int = Field(ge=0)
    criteria_override_explanation: str | None = None

    @model_validator(mode="after")
    def validate_invariants(self) -> LabelDecisionRecord:
        validate_decision_reason(
            decision=self.decision,
            reason_code=self.reason_code,
            decision_source=self.decision_source,
            explanation=self.explanation,
            role=self.role,
        )
        if self.recent_support > self.total_support:
            raise ValueError("recent_support must not exceed total_support")
        if self.active_month_count > self.total_support:
            raise ValueError("active_month_count must not exceed total_support")
        if (self.first_month is None) != (self.last_month is None):
            raise ValueError("first_month and last_month must both be set or unset")
        if self.total_support == 0:
            if self.first_month is not None:
                raise ValueError("a label with zero support must not have month bounds")
        else:
            if self.first_month is None:
                raise ValueError("a label with positive support must have month bounds")
            if self.active_month_count < 1:
                raise ValueError("positive support requires active_month_count >= 1")
        if (
            self.first_month is not None
            and self.last_month is not None
            and self.last_month < self.first_month
        ):
            raise ValueError("last_month must not precede first_month")
        if self.criteria_override_explanation is not None and not (
            self.criteria_override_explanation.strip()
        ):
            raise ValueError("criteria_override_explanation must be non-blank when present")
        return self


class PolicyCoverage(BaseModel):
    """Policy-wide objective coverage metrics over the whole issue population."""

    total_issues: int = Field(ge=0)
    dataset_active_month_count: int = Field(ge=1)
    total_audited_labels: int = Field(ge=0)
    included_label_count: int = Field(ge=0)
    deferred_label_count: int = Field(ge=0)
    excluded_label_count: int = Field(ge=0)
    explicit_label_count: int = Field(ge=0)
    default_label_count: int = Field(ge=0)
    included_labels: list[str] = Field(default_factory=list)
    issues_with_included_target: int = Field(ge=0)
    issues_without_included_target: int = Field(ge=0)
    target_coverage_fraction: float = Field(ge=0.0, le=1.0)
    included_target_assignments: int = Field(ge=0)
    included_target_cardinality: float = Field(ge=0.0)
    included_labels_per_issue: LabelsPerIssueStats

    @model_validator(mode="after")
    def validate_invariants(self) -> PolicyCoverage:
        if (
            self.included_label_count + self.deferred_label_count + self.excluded_label_count
            != self.total_audited_labels
        ):
            raise ValueError(
                "included + deferred + excluded label counts must equal total_audited_labels"
            )
        if self.explicit_label_count + self.default_label_count != self.total_audited_labels:
            raise ValueError(
                "explicit + default label counts must equal total_audited_labels"
            )
        if len(self.included_labels) != self.included_label_count:
            raise ValueError("included_labels length must equal included_label_count")
        if len(set(self.included_labels)) != len(self.included_labels):
            raise ValueError("included_labels must not contain duplicates")
        if (
            self.issues_with_included_target + self.issues_without_included_target
            != self.total_issues
        ):
            raise ValueError(
                "issues_with_included_target + issues_without_included_target must equal "
                "total_issues"
            )
        if self.total_issues > 0:
            expected_coverage = self.issues_with_included_target / self.total_issues
            if not _floats_consistent(self.target_coverage_fraction, expected_coverage):
                raise ValueError(
                    "target_coverage_fraction is inconsistent with issues/total_issues"
                )
            expected_cardinality = self.included_target_assignments / self.total_issues
            if not _floats_consistent(self.included_target_cardinality, expected_cardinality):
                raise ValueError(
                    "included_target_cardinality is inconsistent with assignments/total_issues"
                )
        return self


class LabelPolicyIdentity(BaseModel):
    """Immutable identity and lineage binding a policy to its dataset, audit, and config."""

    policy_version: str
    policy_id: PolicyId
    policy_input_sha256: Sha256Hex
    repository: str
    dataset_id: DatasetId
    dataset_output_sha256: Sha256Hex
    audit_id: AuditId
    audit_json_sha256: Sha256Hex
    audit_version: str
    config_schema_version: str
    config_sha256: Sha256Hex
    issue_schema_version: str
    normalizer_version: str

    @model_validator(mode="after")
    def policy_id_consistent(self) -> LabelPolicyIdentity:
        expected_input = compute_policy_input_sha256(
            policy_version=self.policy_version,
            dataset_id=self.dataset_id,
            dataset_output_sha256=self.dataset_output_sha256,
            audit_id=self.audit_id,
            audit_json_sha256=self.audit_json_sha256,
            config_schema_version=self.config_schema_version,
            config_sha256=self.config_sha256,
        )
        if self.policy_input_sha256 != expected_input:
            raise ValueError(
                "policy_input_sha256 is inconsistent with the policy-input payload"
            )
        expected_id = compute_policy_id(
            self.dataset_id, self.policy_input_sha256, self.policy_version
        )
        if self.policy_id != expected_id:
            raise ValueError(
                f"policy_id {self.policy_id!r} is inconsistent with dataset_id, "
                f"policy_input_sha256, and policy_version (expected {expected_id!r})"
            )
        return self


class LabelPolicyDocument(BaseModel):
    """The full, deterministic, machine-readable policy (the ``label_policy.json`` content).

    The document carries no build timestamp so that its bytes depend only on the dataset,
    the audit, the configuration, and the policy version.
    """

    schema_version: Literal["2"] = LABEL_POLICY_DOCUMENT_SCHEMA_VERSION
    identity: LabelPolicyIdentity
    selection_criteria: SelectionCriteria
    coverage: PolicyCoverage
    decisions: list[LabelDecisionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cross_section(self) -> LabelPolicyDocument:
        if len(self.decisions) != self.coverage.total_audited_labels:
            raise ValueError("decisions length must equal coverage.total_audited_labels")

        names = [record.label for record in self.decisions]
        if len(set(names)) != len(names):
            raise ValueError("decisions must contain exactly one record per label")

        expected_order = sorted(
            self.decisions, key=lambda r: (-r.total_support, r.label)
        )
        if [r.label for r in self.decisions] != [r.label for r in expected_order]:
            raise ValueError("decisions must be ordered by (-total_support, label)")

        included = sorted(r.label for r in self.decisions if r.decision == "include")
        deferred = [r for r in self.decisions if r.decision == "defer"]
        excluded = [r for r in self.decisions if r.decision == "exclude"]
        explicit = [r for r in self.decisions if r.decision_source == "explicit"]
        defaulted = [r for r in self.decisions if r.decision_source == "default"]
        if sorted(self.coverage.included_labels) != included:
            raise ValueError("coverage.included_labels disagrees with include decisions")
        if self.coverage.included_label_count != len(included):
            raise ValueError("coverage.included_label_count disagrees with include decisions")
        if self.coverage.deferred_label_count != len(deferred):
            raise ValueError("coverage.deferred_label_count disagrees with defer decisions")
        if self.coverage.excluded_label_count != len(excluded):
            raise ValueError("coverage.excluded_label_count disagrees with exclude decisions")
        if self.coverage.explicit_label_count != len(explicit):
            raise ValueError("coverage.explicit_label_count disagrees with decision sources")
        if self.coverage.default_label_count != len(defaulted):
            raise ValueError("coverage.default_label_count disagrees with decision sources")

        criteria = self.selection_criteria
        total = self.coverage.total_issues
        for record in self.decisions:
            if record.active_month_count > self.coverage.dataset_active_month_count:
                raise ValueError(
                    f"active_month_count for {record.label!r} exceeds the dataset's "
                    "active-month count"
                )
            if total > 0:
                expected = record.total_support / total
                if not _floats_consistent(record.issue_fraction, expected):
                    raise ValueError(
                        f"issue_fraction for {record.label!r} is inconsistent with "
                        "total_support/total_issues"
                    )
            if record.decision == "include" and record.criteria_override_explanation is None:
                if (
                    record.total_support < criteria.min_total_support
                    or record.active_month_count < criteria.min_active_months
                    or record.recent_support < criteria.min_recent_support
                ):
                    raise ValueError(
                        f"included label {record.label!r} does not satisfy selection "
                        "criteria and carries no override"
                    )
        return self


class LabelPolicyManifest(BaseModel):
    """Validated lineage manifest describing one immutable policy artifact."""

    schema_version: Literal["2"] = LABEL_POLICY_MANIFEST_SCHEMA_VERSION
    label_policy_document_schema_version: Literal["2"] = LABEL_POLICY_DOCUMENT_SCHEMA_VERSION
    policy_version: str
    policy_id: PolicyId
    policy_input_sha256: Sha256Hex
    repository: str
    dataset_id: DatasetId
    dataset_output_sha256: Sha256Hex
    audit_id: AuditId
    audit_json_sha256: Sha256Hex
    audit_version: str
    config_schema_version: str
    config_sha256: Sha256Hex
    issue_schema_version: str
    normalizer_version: str
    built_at: datetime
    total_audited_labels: int = Field(ge=0)
    included_label_count: int = Field(ge=0)
    deferred_label_count: int = Field(ge=0)
    excluded_label_count: int = Field(ge=0)
    explicit_label_count: int = Field(ge=0)
    default_label_count: int = Field(ge=0)
    issues_with_included_target: int = Field(ge=0)
    issues_without_included_target: int = Field(ge=0)
    target_coverage_fraction: float = Field(ge=0.0, le=1.0)
    included_target_assignments: int = Field(ge=0)
    included_target_cardinality: float = Field(ge=0.0)
    label_policy_json_file: str = LABEL_POLICY_JSON_FILE
    label_policy_json_sha256: Sha256Hex
    label_policy_markdown_file: str = LABEL_POLICY_MARKDOWN_FILE
    label_policy_markdown_sha256: Sha256Hex

    @field_validator("built_at")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_serializer("built_at", when_used="json")
    def serialize_built_at(self, value: datetime) -> str:
        return format_utc_datetime(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> LabelPolicyManifest:
        expected_input = compute_policy_input_sha256(
            policy_version=self.policy_version,
            dataset_id=self.dataset_id,
            dataset_output_sha256=self.dataset_output_sha256,
            audit_id=self.audit_id,
            audit_json_sha256=self.audit_json_sha256,
            config_schema_version=self.config_schema_version,
            config_sha256=self.config_sha256,
        )
        if self.policy_input_sha256 != expected_input:
            raise ValueError(
                "policy_input_sha256 is inconsistent with the policy-input payload"
            )
        expected_id = compute_policy_id(
            self.dataset_id, self.policy_input_sha256, self.policy_version
        )
        if self.policy_id != expected_id:
            raise ValueError(
                f"policy_id {self.policy_id!r} is inconsistent with dataset_id, "
                f"policy_input_sha256, and policy_version (expected {expected_id!r})"
            )
        if (
            self.included_label_count + self.deferred_label_count + self.excluded_label_count
            != self.total_audited_labels
        ):
            raise ValueError(
                "included + deferred + excluded label counts must equal total_audited_labels"
            )
        if self.explicit_label_count + self.default_label_count != self.total_audited_labels:
            raise ValueError(
                "explicit + default label counts must equal total_audited_labels"
            )
        if not _is_safe_relative_path(self.label_policy_json_file):
            raise ValueError(
                f"label_policy_json_file must be a safe relative path: "
                f"{self.label_policy_json_file!r}"
            )
        if not _is_safe_relative_path(self.label_policy_markdown_file):
            raise ValueError(
                f"label_policy_markdown_file must be a safe relative path: "
                f"{self.label_policy_markdown_file!r}"
            )
        return self

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)
