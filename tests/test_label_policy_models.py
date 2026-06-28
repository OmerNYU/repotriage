"""Tests for label-policy model invariants, identity, and tolerant float reconciliation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from repotriage.label_policy.models import (
    LabelDecisionRecord,
    LabelPolicyDocument,
    LabelPolicyIdentity,
    LabelsPerIssueStats,
    PolicyCoverage,
    SelectionCriteria,
    compute_policy_id,
    compute_policy_input_sha256,
)

_DATASET_ID = "20260628T161306010651Z-n1-074402d21505"
_CONFIG_SHA = "a" * 64
_AUDIT_ID = f"{_DATASET_ID}-a2"
_AUDIT_JSON_SHA = "b" * 64
_DATASET_OUTPUT_SHA = "c" * 64


def _input_sha(**overrides: str) -> str:
    fields = {
        "policy_version": "2",
        "dataset_id": _DATASET_ID,
        "dataset_output_sha256": _DATASET_OUTPUT_SHA,
        "audit_id": _AUDIT_ID,
        "audit_json_sha256": _AUDIT_JSON_SHA,
        "config_schema_version": "2",
        "config_sha256": _CONFIG_SHA,
    }
    fields.update(overrides)
    return compute_policy_input_sha256(**fields)  # type: ignore[arg-type]


_POLICY_INPUT_SHA = _input_sha()
_POLICY_ID = compute_policy_id(_DATASET_ID, _POLICY_INPUT_SHA, "2")

_CRITERIA = SelectionCriteria(
    min_total_support=1, min_active_months=1, min_recent_support=1, recent_window_months=2
)


def _record(label: str, **overrides: object) -> LabelDecisionRecord:
    fields: dict[str, object] = {
        "label": label,
        "decision": "include",
        "decision_source": "explicit",
        "role": "issue_type",
        "leakage_risk": "low",
        "reason_code": "selected_target",
        "explanation": "x",
        "total_support": 4,
        "issue_fraction": 0.4,
        "active_month_count": 1,
        "first_month": "2025-01",
        "last_month": "2025-01",
        "recent_support": 4,
    }
    fields.update(overrides)
    return LabelDecisionRecord(**fields)  # type: ignore[arg-type]


def _default_excluded(label: str = "Misc") -> LabelDecisionRecord:
    return LabelDecisionRecord(
        label=label,
        decision="exclude",
        decision_source="default",
        role="unreviewed",
        leakage_risk="high",
        reason_code="unreviewed_default",
        explanation="d",
        total_support=0,
        issue_fraction=0.0,
        active_month_count=0,
        first_month=None,
        last_month=None,
        recent_support=0,
    )


def test_record_rejects_recent_exceeding_total() -> None:
    with pytest.raises(ValidationError, match="recent_support"):
        _record("Bug", total_support=2, recent_support=3)


def test_record_requires_both_month_bounds() -> None:
    with pytest.raises(ValidationError, match="first_month and last_month"):
        _record("Bug", last_month=None)


def test_record_positive_support_requires_months() -> None:
    with pytest.raises(ValidationError, match="positive support"):
        _record("Bug", first_month=None, last_month=None, active_month_count=0, recent_support=0)


def test_record_default_source_consistency_enforced() -> None:
    with pytest.raises(ValidationError, match="default-sourced"):
        _record(
            "Misc",
            decision="exclude",
            decision_source="default",
            role="unreviewed",
            reason_code="workflow_label",
            total_support=0,
            issue_fraction=0.0,
            active_month_count=0,
            first_month=None,
            last_month=None,
            recent_support=0,
        )


def _coverage(**overrides: object) -> PolicyCoverage:
    fields: dict[str, object] = {
        "total_issues": 10,
        "dataset_active_month_count": 3,
        "total_audited_labels": 2,
        "included_label_count": 1,
        "deferred_label_count": 0,
        "excluded_label_count": 1,
        "explicit_label_count": 1,
        "default_label_count": 1,
        "included_labels": ["Bug"],
        "issues_with_included_target": 4,
        "issues_without_included_target": 6,
        "target_coverage_fraction": 0.4,
        "included_target_assignments": 4,
        "included_target_cardinality": 0.4,
        "included_labels_per_issue": LabelsPerIssueStats(min=0, median=0.0, mean=0.4, max=1),
    }
    fields.update(overrides)
    return PolicyCoverage(**fields)  # type: ignore[arg-type]


def test_coverage_decision_counts_must_sum() -> None:
    with pytest.raises(ValidationError, match="must equal total_audited_labels"):
        _coverage(excluded_label_count=5)


def test_coverage_source_counts_must_sum() -> None:
    with pytest.raises(ValidationError, match="explicit \\+ default"):
        _coverage(default_label_count=5)


def test_coverage_accepts_tiny_float_perturbation() -> None:
    coverage = _coverage(target_coverage_fraction=0.4 + 1e-15)
    assert coverage.issues_with_included_target == 4


def test_coverage_rejects_material_fraction_error() -> None:
    with pytest.raises(ValidationError, match="target_coverage_fraction"):
        _coverage(target_coverage_fraction=0.5)


def test_input_sha_changes_with_each_field() -> None:
    base = _input_sha()
    assert _input_sha(dataset_id="20260628T161306010651Z-n1-ffffffffffff") != base
    assert _input_sha(dataset_output_sha256="d" * 64) != base
    assert _input_sha(audit_id=f"{_DATASET_ID}-a3") != base
    assert _input_sha(audit_json_sha256="e" * 64) != base
    assert _input_sha(config_schema_version="3") != base
    assert _input_sha(config_sha256="f" * 64) != base
    assert _input_sha(policy_version="3") != base


def test_identity_rejects_inconsistent_policy_id() -> None:
    with pytest.raises(ValidationError, match="policy_id"):
        _identity(policy_id=f"{_DATASET_ID}-lp2-ffffffffffff")


def test_identity_rejects_inconsistent_input_sha() -> None:
    with pytest.raises(ValidationError, match="policy_input_sha256"):
        _identity(policy_input_sha256="0" * 64)


def _identity(**overrides: object) -> LabelPolicyIdentity:
    fields: dict[str, object] = {
        "policy_version": "2",
        "policy_id": _POLICY_ID,
        "policy_input_sha256": _POLICY_INPUT_SHA,
        "repository": "pandas-dev/pandas",
        "dataset_id": _DATASET_ID,
        "dataset_output_sha256": _DATASET_OUTPUT_SHA,
        "audit_id": _AUDIT_ID,
        "audit_json_sha256": _AUDIT_JSON_SHA,
        "audit_version": "2",
        "config_schema_version": "2",
        "config_sha256": _CONFIG_SHA,
        "issue_schema_version": "1",
        "normalizer_version": "1",
    }
    fields.update(overrides)
    return LabelPolicyIdentity(**fields)  # type: ignore[arg-type]


def test_document_requires_one_decision_per_audited_label() -> None:
    with pytest.raises(ValidationError, match="decisions length"):
        LabelPolicyDocument(
            identity=_identity(),
            selection_criteria=_CRITERIA,
            coverage=_coverage(),
            decisions=[_record("Bug")],
        )


def test_document_included_consistency() -> None:
    document = LabelPolicyDocument(
        identity=_identity(),
        selection_criteria=_CRITERIA,
        coverage=_coverage(),
        decisions=[_record("Bug"), _default_excluded()],
    )
    assert document.coverage.included_labels == ["Bug"]


def test_document_rejects_included_below_criteria() -> None:
    criteria = SelectionCriteria(
        min_total_support=99, min_active_months=1, min_recent_support=1, recent_window_months=2
    )
    with pytest.raises(ValidationError, match="selection criteria"):
        LabelPolicyDocument(
            identity=_identity(),
            selection_criteria=criteria,
            coverage=_coverage(),
            decisions=[_record("Bug"), _default_excluded()],
        )
