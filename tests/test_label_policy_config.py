"""Tests for label-policy configuration parsing, validation, and canonical hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotriage.label_policy.config import config_sha256, load_config
from repotriage.label_policy.models import LabelPolicyConfigError
from tests.helpers import write_label_policy_config


def _entry(label: str, **overrides: object) -> dict[str, object]:
    entry = {
        "label": label,
        "decision": "include",
        "role": "issue_type",
        "leakage_risk": "low",
        "reason_code": "selected_target",
        "explanation": f"{label} explanation",
    }
    entry.update(overrides)
    return entry


def test_valid_config_parses(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug"), _entry("Docs", role="component")],
        selection_criteria={
            "min_total_support": 30,
            "min_active_months": 12,
            "min_recent_support": 5,
            "recent_window_months": 4,
        },
        notes="reviewed",
    )

    config, digest = load_config(path)

    assert config.config_schema_version == "2"
    assert config.repository == "pandas-dev/pandas"
    assert config.selection_criteria.min_total_support == 30
    assert config.selection_criteria.recent_window_months == 4
    assert len(digest) == 64


def test_duplicate_entries_rejected(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug"), _entry("Bug", decision="exclude", role="workflow",
                                      reason_code="workflow_label")],
    )
    with pytest.raises(LabelPolicyConfigError, match="duplicate"):
        load_config(path)


def test_default_must_be_safe(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug")],
        default={
            "decision": "include",
            "role": "unreviewed",
            "reason_code": "unreviewed_default",
            "leakage_risk": "high",
            "explanation": "bad default",
        },
    )
    with pytest.raises(LabelPolicyConfigError, match="default decision"):
        load_config(path)


def test_default_reason_must_be_unreviewed_default(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug")],
        default={
            "decision": "exclude",
            "role": "unreviewed",
            "reason_code": "workflow_label",
            "leakage_risk": "high",
            "explanation": "bad reason",
        },
    )
    with pytest.raises(LabelPolicyConfigError, match="unreviewed_default"):
        load_config(path)


def test_selection_criteria_required(tmp_path: Path) -> None:
    payload = {
        "config_schema_version": "2",
        "repository": "pandas-dev/pandas",
        "notes": "",
        "default": {
            "decision": "exclude",
            "role": "unreviewed",
            "reason_code": "unreviewed_default",
            "leakage_risk": "high",
            "explanation": "d",
        },
        "labels": [],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelPolicyConfigError):
        load_config(path)


def test_selection_criteria_must_be_positive(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug")],
        selection_criteria={
            "min_total_support": 0,
            "min_active_months": 12,
            "min_recent_support": 5,
            "recent_window_months": 4,
        },
    )
    with pytest.raises(LabelPolicyConfigError):
        load_config(path)


def test_unknown_enum_value_rejected(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json", labels=[_entry("Bug", role="not-a-role")]
    )
    with pytest.raises(LabelPolicyConfigError):
        load_config(path)


def test_extra_field_rejected(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json", labels=[_entry("Bug", surprise="x")]
    )
    with pytest.raises(LabelPolicyConfigError):
        load_config(path)


def test_include_requires_selected_target(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug", reason_code="workflow_label")],
    )
    with pytest.raises(LabelPolicyConfigError, match="selected_target"):
        load_config(path)


def test_explicit_exclude_requires_semantic_reason(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Needs Triage", decision="exclude", role="workflow",
                       reason_code="selected_target")],
    )
    with pytest.raises(LabelPolicyConfigError, match="workflow_label"):
        load_config(path)


def test_manual_deferral_requires_nonblank_explanation(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Groupby", decision="defer", role="component",
                       reason_code="manual_deferral", explanation=" ")],
    )
    with pytest.raises(LabelPolicyConfigError, match="manual_deferral"):
        load_config(path)


def test_reserved_unreviewed_default_rejected_for_explicit(tmp_path: Path) -> None:
    path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[_entry("Bug", decision="exclude", role="workflow",
                       reason_code="unreviewed_default")],
    )
    with pytest.raises(LabelPolicyConfigError, match="unreviewed_default"):
        load_config(path)


def test_hash_independent_of_whitespace_key_and_label_order(tmp_path: Path) -> None:
    pretty = write_label_policy_config(
        tmp_path / "pretty.json",
        labels=[_entry("Bug"), _entry("Docs", role="component")],
        selection_criteria={
            "min_total_support": 1,
            "min_active_months": 1,
            "min_recent_support": 1,
            "recent_window_months": 4,
        },
        indent=2,
    )
    # Same semantics, compact, label entries swapped and keys reordered.
    swapped = {
        "labels": [
            {
                "explanation": "Docs explanation",
                "label": "Docs",
                "role": "component",
                "decision": "include",
                "reason_code": "selected_target",
                "leakage_risk": "low",
            },
            {
                "reason_code": "selected_target",
                "explanation": "Bug explanation",
                "decision": "include",
                "role": "issue_type",
                "leakage_risk": "low",
                "label": "Bug",
            },
        ],
        "default": {
            "explanation": "default exclusion",
            "decision": "exclude",
            "leakage_risk": "high",
            "reason_code": "unreviewed_default",
            "role": "unreviewed",
        },
        "selection_criteria": {
            "recent_window_months": 4,
            "min_recent_support": 1,
            "min_active_months": 1,
            "min_total_support": 1,
        },
        "notes": "",
        "repository": "pandas-dev/pandas",
        "config_schema_version": "2",
    }
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(swapped, separators=(",", ":")), encoding="utf-8")

    _, pretty_hash = load_config(pretty)
    _, compact_hash = load_config(compact)
    assert pretty_hash == compact_hash


def test_changed_decision_changes_hash(tmp_path: Path) -> None:
    config_a, hash_a = load_config(
        write_label_policy_config(tmp_path / "a.json", labels=[_entry("Bug")])
    )
    deferred = _entry("Bug", decision="defer", reason_code="manual_deferral",
                      explanation="deferred manually")
    _, hash_b = load_config(
        write_label_policy_config(tmp_path / "b.json", labels=[deferred])
    )
    assert hash_a != hash_b
    assert config_sha256(config_a) == hash_a


def test_changed_criteria_changes_hash(tmp_path: Path) -> None:
    _, hash_a = load_config(
        write_label_policy_config(
            tmp_path / "a.json",
            labels=[_entry("Bug")],
            selection_criteria={
                "min_total_support": 1, "min_active_months": 1,
                "min_recent_support": 1, "recent_window_months": 4,
            },
        )
    )
    _, hash_b = load_config(
        write_label_policy_config(
            tmp_path / "b.json",
            labels=[_entry("Bug")],
            selection_criteria={
                "min_total_support": 2, "min_active_months": 1,
                "min_recent_support": 1, "recent_window_months": 4,
            },
        )
    )
    assert hash_a != hash_b
