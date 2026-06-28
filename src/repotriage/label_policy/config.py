"""Human-authored label-decision configuration: schema, loading, and canonical hashing.

The configuration is the single source of truth for which labels are first-model targets.
Decisions are explicit; semantic roles and reasons are never inferred from label names. The
canonical configuration hash is computed from the parsed, validated, and label-sorted model
(not incidental whitespace, JSON key order, or label-entry order), so reformatting or
reordering entries does not change policy identity while changing any meaningful field does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from repotriage.label_policy.models import (
    Decision,
    LabelPolicyConfigError,
    LeakageRisk,
    ReasonCode,
    Role,
    SelectionCriteria,
    validate_decision_reason,
)

CONFIG_SCHEMA_VERSION: Literal["2"] = "2"


class DefaultDecision(BaseModel):
    """The single safe default applied to every audited label not explicitly listed."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    role: Role
    reason_code: ReasonCode
    leakage_risk: LeakageRisk = "high"
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def must_be_safe_default(self) -> DefaultDecision:
        if self.decision != "exclude":
            raise ValueError("default decision must be 'exclude'; include/defer must be explicit")
        if self.role != "unreviewed":
            raise ValueError("default role must be 'unreviewed'")
        if self.reason_code != "unreviewed_default":
            raise ValueError("default reason_code must be 'unreviewed_default'")
        return self


class LabelDecisionEntry(BaseModel):
    """One explicitly reviewed label decision in the human-authored configuration."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    decision: Decision
    role: Role
    leakage_risk: LeakageRisk
    reason_code: ReasonCode
    explanation: str = Field(min_length=1)
    criteria_override_explanation: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> LabelDecisionEntry:
        validate_decision_reason(
            decision=self.decision,
            reason_code=self.reason_code,
            decision_source="explicit",
            explanation=self.explanation,
            role=self.role,
        )
        if self.criteria_override_explanation is not None:
            if not self.criteria_override_explanation.strip():
                raise ValueError("criteria_override_explanation must be non-blank when present")
            if self.decision != "include":
                raise ValueError("criteria_override_explanation is only valid for include")
        return self


class LabelPolicyConfig(BaseModel):
    """Validated human-authored configuration for one repository's label policy."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["2"] = CONFIG_SCHEMA_VERSION
    repository: str = Field(min_length=1)
    notes: str = ""
    selection_criteria: SelectionCriteria
    default: DefaultDecision
    labels: list[LabelDecisionEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_invariants(self) -> LabelPolicyConfig:
        names = [entry.label for entry in self.labels]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            listed = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate label entries in configuration: {listed}")
        return self


def canonical_config_bytes(config: LabelPolicyConfig) -> bytes:
    """Return the canonical UTF-8 bytes of a validated configuration.

    The label list is sorted by name before serialization so that swapping two
    semantically identical entries in the source file does not change the hash. The result
    is compact JSON with sorted keys, independent of source whitespace and key ordering.
    """
    payload = config.model_dump(mode="json")
    payload["labels"] = sorted(payload["labels"], key=lambda entry: entry["label"])
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def config_sha256(config: LabelPolicyConfig) -> str:
    """Compute the canonical SHA-256 hex digest of a validated configuration."""
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def load_config(config_path: Path) -> tuple[LabelPolicyConfig, str]:
    """Load, validate, and hash a configuration file.

    Returns the validated model and its canonical SHA-256. Any read, JSON, or schema
    failure is reported as :class:`LabelPolicyConfigError`.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelPolicyConfigError(
            f"Unable to read label-policy configuration at {config_path}: {exc}"
        ) from exc
    try:
        config = LabelPolicyConfig.model_validate_json(text)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LabelPolicyConfigError(
            f"Invalid label-policy configuration at {config_path}: {exc}"
        ) from exc
    return config, config_sha256(config)
