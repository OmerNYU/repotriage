"""Objective analysis for the label policy.

Decisions (include/defer/exclude) are resolved purely from the human-authored
configuration and the audited label set. Objective enrichment facts (support, active
months, recent-window support, coverage) are derived by re-streaming the normalized
dataset, because the audit document does not carry per-label monthly support.
Dataset-derived counts are cross-checked against the audit so the two artifacts can never
silently disagree. Every included label must satisfy the configured selection criteria
(inclusive thresholds) unless it carries an explicit override explanation.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from repotriage.audit.models import AuditDocument
from repotriage.audit.reader import read_dataset_issues
from repotriage.dataset.models import ProcessedManifest
from repotriage.label_policy.config import LabelDecisionEntry, LabelPolicyConfig
from repotriage.label_policy.models import (
    LabelDecisionRecord,
    LabelPolicyConfigError,
    LabelPolicyInputError,
    LabelsPerIssueStats,
    PolicyCoverage,
)


def _percentile(sorted_values: list[int], percentile: float) -> float:
    """Type-7 linear-interpolation percentile over an ascending, non-empty sample."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile requires a non-empty sample")
    if n == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (n - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    frac = rank - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


@dataclass(frozen=True)
class LabelPolicyAnalysis:
    """The objective sections produced for a policy document."""

    coverage: PolicyCoverage
    decisions: list[LabelDecisionRecord]


@dataclass(frozen=True)
class _ResolvedDecision:
    decision: str
    decision_source: str
    role: str
    leakage_risk: str
    reason_code: str
    explanation: str
    criteria_override_explanation: str | None


def _resolve_decision(label: str, entries: dict[str, LabelDecisionEntry],
                      config: LabelPolicyConfig) -> _ResolvedDecision:
    entry = entries.get(label)
    if entry is not None:
        return _ResolvedDecision(
            decision=entry.decision,
            decision_source="explicit",
            role=entry.role,
            leakage_risk=entry.leakage_risk,
            reason_code=entry.reason_code,
            explanation=entry.explanation,
            criteria_override_explanation=entry.criteria_override_explanation,
        )
    default = config.default
    return _ResolvedDecision(
        decision=default.decision,
        decision_source="default",
        role=default.role,
        leakage_risk=default.leakage_risk,
        reason_code=default.reason_code,
        explanation=default.explanation,
        criteria_override_explanation=None,
    )


def analyze_label_policy(
    dataset_dir: Path,
    processed_manifest: ProcessedManifest,
    audit_document: AuditDocument,
    config: LabelPolicyConfig,
) -> LabelPolicyAnalysis:
    """Resolve decisions and compute objective coverage, cross-checking against the audit."""
    audited_labels = [label.name for label in audit_document.label_metrics.labels]
    audited_set = set(audited_labels)
    audit_counts = {label.name: label.count for label in audit_document.label_metrics.labels}
    entries = {entry.label: entry for entry in config.labels}

    unknown = sorted(label for label in entries if label not in audited_set)
    if unknown:
        listed = ", ".join(repr(name) for name in unknown)
        raise LabelPolicyConfigError(
            f"Configuration lists labels not present in the audit: {listed}."
        )

    resolved = {label: _resolve_decision(label, entries, config) for label in audited_labels}
    included_set = {label for label, r in resolved.items() if r.decision == "include"}

    total_issues = 0
    issues_by_month: Counter[str] = Counter()
    label_total: Counter[str] = Counter()
    label_months: defaultdict[str, set[str]] = defaultdict(set)
    label_month_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    included_per_issue: list[int] = []

    for issue in read_dataset_issues(dataset_dir, processed_manifest):
        total_issues += 1
        month = issue.created_at.strftime("%Y-%m")
        issues_by_month[month] += 1

        labels = sorted(set(issue.labels))
        for label in labels:
            label_total[label] += 1
            label_months[label].add(month)
            label_month_counts[label][month] += 1

        included_per_issue.append(sum(1 for label in labels if label in included_set))

    _cross_check_against_audit(
        total_issues=total_issues,
        issues_by_month=dict(issues_by_month),
        label_total=label_total,
        audit_document=audit_document,
        audit_counts=audit_counts,
        audited_set=audited_set,
    )

    # Recent active-month window: sort distinct active YYYY-MM dataset keys, take the final
    # recent_window_months keys (all of them when fewer exist).
    window_months = config.selection_criteria.recent_window_months
    active_months_sorted = sorted(issues_by_month)
    recent_window = set(active_months_sorted[-window_months:])

    decisions: list[LabelDecisionRecord] = []
    for label in audited_labels:
        r = resolved[label]
        months = label_months.get(label, set())
        total_support = label_total.get(label, 0)
        recent_support = sum(
            count for month, count in label_month_counts.get(label, Counter()).items()
            if month in recent_window
        )
        decisions.append(
            LabelDecisionRecord(
                label=label,
                decision=r.decision,
                decision_source=r.decision_source,
                role=r.role,
                leakage_risk=r.leakage_risk,
                reason_code=r.reason_code,
                explanation=r.explanation,
                total_support=total_support,
                issue_fraction=(total_support / total_issues) if total_issues else 0.0,
                active_month_count=len(months),
                first_month=min(months) if months else None,
                last_month=max(months) if months else None,
                recent_support=recent_support,
                criteria_override_explanation=r.criteria_override_explanation,
            )
        )

    _enforce_selection_criteria(decisions, config)

    decisions.sort(key=lambda record: (-record.total_support, record.label))
    coverage = _build_coverage(
        total_issues=total_issues,
        dataset_active_month_count=len(issues_by_month),
        audited_labels=audited_labels,
        decisions=decisions,
        included_per_issue=included_per_issue,
    )
    return LabelPolicyAnalysis(coverage=coverage, decisions=decisions)


def _enforce_selection_criteria(
    decisions: list[LabelDecisionRecord], config: LabelPolicyConfig
) -> None:
    """Reject any included label that fails a threshold without an explicit override."""
    criteria = config.selection_criteria
    for record in decisions:
        if record.decision != "include" or record.criteria_override_explanation is not None:
            continue
        failures: list[str] = []
        if record.total_support < criteria.min_total_support:
            failures.append(
                f"total_support {record.total_support} < {criteria.min_total_support}"
            )
        if record.active_month_count < criteria.min_active_months:
            failures.append(
                f"active_month_count {record.active_month_count} < {criteria.min_active_months}"
            )
        if record.recent_support < criteria.min_recent_support:
            failures.append(
                f"recent_support {record.recent_support} < {criteria.min_recent_support}"
            )
        if failures:
            listed = "; ".join(failures)
            raise LabelPolicyConfigError(
                f"Included label {record.label!r} violates selection criteria ({listed}) "
                "and carries no criteria_override_explanation."
            )


def _cross_check_against_audit(
    *,
    total_issues: int,
    issues_by_month: dict[str, int],
    label_total: Counter[str],
    audit_document: AuditDocument,
    audit_counts: dict[str, int],
    audited_set: set[str],
) -> None:
    """Verify dataset-derived counts agree with the audit; raise on any disagreement."""
    expected_total = audit_document.repository_summary.total_issues
    if total_issues != expected_total:
        raise LabelPolicyInputError(
            f"Dataset issue count {total_issues} disagrees with audit total {expected_total}."
        )

    audit_months = dict(audit_document.temporal_metrics.issues_by_month)
    if issues_by_month != audit_months:
        raise LabelPolicyInputError(
            "Dataset monthly issue counts disagree with the audit temporal metrics."
        )

    dataset_label_set = set(label_total)
    if dataset_label_set != audited_set:
        missing = sorted(audited_set - dataset_label_set)
        extra = sorted(dataset_label_set - audited_set)
        raise LabelPolicyInputError(
            "Dataset label set disagrees with the audit label set "
            f"(missing={missing}, unexpected={extra})."
        )

    for label, expected_count in audit_counts.items():
        actual = label_total.get(label, 0)
        if actual != expected_count:
            raise LabelPolicyInputError(
                f"Dataset support for label {label!r} is {actual} but the audit records "
                f"{expected_count}."
            )


def _build_coverage(
    *,
    total_issues: int,
    dataset_active_month_count: int,
    audited_labels: list[str],
    decisions: list[LabelDecisionRecord],
    included_per_issue: list[int],
) -> PolicyCoverage:
    included = [r for r in decisions if r.decision == "include"]
    deferred = [r for r in decisions if r.decision == "defer"]
    excluded = [r for r in decisions if r.decision == "exclude"]
    explicit = [r for r in decisions if r.decision_source == "explicit"]
    defaulted = [r for r in decisions if r.decision_source == "default"]

    included_labels = [r.label for r in included]  # already sorted by (-support, name)

    issues_with = sum(1 for count in included_per_issue if count >= 1)
    issues_without = total_issues - issues_with
    assignments = sum(included_per_issue)

    if included_per_issue:
        ordered = sorted(included_per_issue)
        per_issue_stats = LabelsPerIssueStats(
            min=ordered[0],
            median=_percentile(ordered, 50),
            mean=sum(ordered) / len(ordered),
            max=ordered[-1],
        )
    else:
        per_issue_stats = LabelsPerIssueStats(min=0, median=0.0, mean=0.0, max=0)

    return PolicyCoverage(
        total_issues=total_issues,
        dataset_active_month_count=dataset_active_month_count,
        total_audited_labels=len(audited_labels),
        included_label_count=len(included),
        deferred_label_count=len(deferred),
        excluded_label_count=len(excluded),
        explicit_label_count=len(explicit),
        default_label_count=len(defaulted),
        included_labels=included_labels,
        issues_with_included_target=issues_with,
        issues_without_included_target=issues_without,
        target_coverage_fraction=(issues_with / total_issues) if total_issues else 0.0,
        included_target_assignments=assignments,
        included_target_cardinality=(assignments / total_issues) if total_issues else 0.0,
        included_labels_per_issue=per_issue_stats,
    )
