"""Tests for label-policy analysis: decisions, temporal stats, coverage, cross-check."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from repotriage.audit.builder import audit_dataset
from repotriage.audit.models import AuditDocument
from repotriage.dataset.models import ProcessedManifest
from repotriage.github.models import RepositoryRef
from repotriage.label_policy.analyzer import analyze_label_policy
from repotriage.label_policy.config import load_config
from repotriage.label_policy.models import LabelPolicyConfigError, LabelPolicyInputError
from tests.helpers import (
    make_normalized_issue,
    write_label_policy_config,
    write_processed_dataset,
)

# Recent active-month window of 2 -> the final two active months (2025-05, 2025-06).
_CRITERIA = {
    "min_total_support": 2,
    "min_active_months": 2,
    "min_recent_support": 1,
    "recent_window_months": 2,
}


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def _standard_issues() -> list:
    return [
        make_normalized_issue(1, labels=["Bug"], created_at=_at(2025, 1, 5)),
        make_normalized_issue(2, labels=["Bug", "Docs"], created_at=_at(2025, 5, 5)),
        make_normalized_issue(3, labels=["Docs"], created_at=_at(2025, 6, 5)),
        make_normalized_issue(4, labels=["Groupby"], created_at=_at(2025, 3, 5)),
        make_normalized_issue(5, labels=["Needs Triage"], created_at=_at(2025, 4, 5)),
        make_normalized_issue(6, labels=["Misc"], created_at=_at(2025, 2, 5)),
    ]


def _config_labels() -> list[dict[str, object]]:
    return [
        {
            "label": "Bug",
            "decision": "include",
            "role": "issue_type",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "bug",
        },
        {
            "label": "Docs",
            "decision": "include",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "docs",
        },
        {
            "label": "Groupby",
            "decision": "defer",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "insufficient_recent_support",
            "explanation": "groupby",
        },
        {
            "label": "Needs Triage",
            "decision": "exclude",
            "role": "workflow",
            "leakage_risk": "high",
            "reason_code": "workflow_label",
            "explanation": "workflow",
        },
    ]


def _prepare(tmp_path: Path, issues: list) -> tuple[Path, ProcessedManifest, AuditDocument]:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    dataset_dir, dataset_id = write_processed_dataset(processed_root, repository, issues)
    manifest = ProcessedManifest.model_validate_json(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    audit = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    return dataset_dir, manifest, audit.document


def _config(tmp_path: Path, labels: list[dict[str, object]] | None = None):
    return load_config(
        write_label_policy_config(
            tmp_path / "policy.json",
            labels=labels if labels is not None else _config_labels(),
            selection_criteria=_CRITERIA,
        )
    )[0]


def test_decisions_cover_every_audited_label(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    analysis = analyze_label_policy(dataset_dir, manifest, audit_document, _config(tmp_path))

    labels = {record.label for record in analysis.decisions}
    assert labels == {"Bug", "Docs", "Groupby", "Needs Triage", "Misc"}
    assert len(analysis.decisions) == audit_document.label_metrics.unique_label_count


def test_default_applied_to_unlisted_label(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    analysis = analyze_label_policy(dataset_dir, manifest, audit_document, _config(tmp_path))

    misc = next(r for r in analysis.decisions if r.label == "Misc")
    assert (misc.decision, misc.decision_source, misc.role, misc.reason_code) == (
        "exclude",
        "default",
        "unreviewed",
        "unreviewed_default",
    )
    bug = next(r for r in analysis.decisions if r.label == "Bug")
    assert bug.decision_source == "explicit"


def test_support_copied_and_temporal_stats(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    analysis = analyze_label_policy(dataset_dir, manifest, audit_document, _config(tmp_path))

    bug = next(r for r in analysis.decisions if r.label == "Bug")
    docs = next(r for r in analysis.decisions if r.label == "Docs")
    groupby = next(r for r in analysis.decisions if r.label == "Groupby")

    assert bug.total_support == 2
    assert bug.active_month_count == 2
    assert (bug.first_month, bug.last_month) == ("2025-01", "2025-05")
    assert bug.recent_support == 1  # recent window is 2025-05..2025-06
    assert docs.recent_support == 2
    assert groupby.recent_support == 0

    audit_counts = {label.name: label.count for label in audit_document.label_metrics.labels}
    for record in analysis.decisions:
        assert record.total_support == audit_counts[record.label]


def test_coverage_metrics(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    coverage = analyze_label_policy(
        dataset_dir, manifest, audit_document, _config(tmp_path)
    ).coverage

    assert coverage.total_issues == 6
    assert coverage.dataset_active_month_count == 6
    assert coverage.total_audited_labels == 5
    assert coverage.included_label_count == 2
    assert coverage.deferred_label_count == 1
    assert coverage.excluded_label_count == 2
    assert coverage.explicit_label_count == 4
    assert coverage.default_label_count == 1
    assert coverage.included_labels == ["Bug", "Docs"]
    assert coverage.issues_with_included_target == 3
    assert coverage.issues_without_included_target == 3
    assert coverage.included_target_assignments == 4
    assert coverage.included_target_cardinality == pytest.approx(4 / 6)
    assert coverage.target_coverage_fraction == pytest.approx(0.5)


def test_unknown_config_label_rejected(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    labels = _config_labels()
    labels.append(
        {
            "label": "Nonexistent",
            "decision": "include",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "missing",
        }
    )
    config = _config(tmp_path, labels)
    with pytest.raises(LabelPolicyConfigError, match="not present in the audit"):
        analyze_label_policy(dataset_dir, manifest, audit_document, config)


def test_dataset_audit_cross_check_rejects_mismatch(tmp_path: Path) -> None:
    dataset_dir, manifest, _ = _prepare(tmp_path, _standard_issues())
    other_issues = _standard_issues()
    other_issues.append(make_normalized_issue(7, labels=["Bug"], created_at=_at(2025, 6, 6)))
    _, _, other_audit_document = _prepare(tmp_path / "other", other_issues)
    with pytest.raises(LabelPolicyInputError):
        analyze_label_policy(dataset_dir, manifest, other_audit_document, _config(tmp_path))


def test_included_label_below_threshold_rejected(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    # Groupby has total_support 1 and recent_support 0; including it must fail enforcement.
    labels = _config_labels()
    labels[2] = {
        "label": "Groupby",
        "decision": "include",
        "role": "component",
        "leakage_risk": "low",
        "reason_code": "selected_target",
        "explanation": "forced include",
    }
    config = _config(tmp_path, labels)
    with pytest.raises(LabelPolicyConfigError, match="violates selection criteria"):
        analyze_label_policy(dataset_dir, manifest, audit_document, config)


def test_included_label_below_threshold_allowed_with_override(tmp_path: Path) -> None:
    dataset_dir, manifest, audit_document = _prepare(tmp_path, _standard_issues())
    labels = _config_labels()
    labels[2] = {
        "label": "Groupby",
        "decision": "include",
        "role": "component",
        "leakage_risk": "low",
        "reason_code": "selected_target",
        "explanation": "forced include",
        "criteria_override_explanation": "manually approved despite low support",
    }
    config = _config(tmp_path, labels)
    analysis = analyze_label_policy(dataset_dir, manifest, audit_document, config)
    groupby = next(r for r in analysis.decisions if r.label == "Groupby")
    assert groupby.decision == "include"
    assert groupby.criteria_override_explanation is not None
