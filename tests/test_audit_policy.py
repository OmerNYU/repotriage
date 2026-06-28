"""Tests for the heuristic suitability policy and its boundary semantics."""

from __future__ import annotations

from datetime import UTC, datetime

from repotriage.audit.analyzer import AuditAnalysis
from repotriage.audit.models import (
    CountFraction,
    LabelFrequency,
    LabelMetrics,
    LabelsPerIssueStats,
    RareLabelBuckets,
    RepositorySummary,
    TemporalMetrics,
    TextFieldStats,
    TextMetrics,
    TextStructuralIndicators,
)
from repotriage.audit.policy import build_warnings

_DT = datetime(2025, 3, 1, 12, 0, 0, tzinfo=UTC)


def _month_keys(count: int) -> dict[str, int]:
    return {f"{2000 + i // 12}-{i % 12 + 1:02d}": 1 for i in range(count)}


def _analysis(
    *,
    total: int = 1000,
    labelled: int = 1000,
    months: int = 12,
    unique_labels: int = 10,
    lt_10: int = 0,
    short_count: int = 0,
) -> AuditAnalysis:
    unlabelled = total - labelled
    months_map = _month_keys(months)
    summary = RepositorySummary(
        total_issues=total,
        labelled_issues=labelled,
        unlabelled_issues=unlabelled,
        labelled_fraction=labelled / total if total else 0.0,
        unlabelled_fraction=unlabelled / total if total else 0.0,
        open_issues=total,
        closed_issues=0,
        null_author_issues=0,
        earliest_created_at=None,
        latest_created_at=None,
        temporal_span_days=0.0,
        active_month_count=months,
        calendar_span_months=months,
    )
    labels = LabelMetrics(
        unique_label_count=unique_labels,
        total_label_assignments=unique_labels,
        zero_label_issue_count=unlabelled,
        labels_per_issue=LabelsPerIssueStats(min=0, median=0.0, mean=0.0, max=0),
        label_cardinality=0.0,
        label_density=0.0,
        rare_label_buckets=RareLabelBuckets(
            lt_5=0, lt_10=lt_10, lt_20=0, lt_50=0, lt_100=0
        ),
        labels=[
            LabelFrequency(
                name=f"label-{i}",
                count=1,
                fraction=1 / total if total else 0.0,
                first_created_at=_DT,
                last_created_at=_DT,
            )
            for i in range(unique_labels)
        ],
        label_pairs=[],
    )
    structural = TextStructuralIndicators(
        empty_bodies=CountFraction(count=0, fraction=0.0),
        short_bodies_lt_100=CountFraction(
            count=short_count, fraction=short_count / total if total else 0.0
        ),
        long_bodies_gt_10000=CountFraction(count=0, fraction=0.0),
        with_code_fence=CountFraction(count=0, fraction=0.0),
        with_url=CountFraction(count=0, fraction=0.0),
        with_heading=CountFraction(count=0, fraction=0.0),
    )
    text = TextMetrics(
        title_chars=TextFieldStats(),
        body_chars=TextFieldStats(),
        total_text_chars=TextFieldStats(),
        structural=structural,
    )
    temporal = TemporalMetrics(
        earliest_created_at=None,
        latest_created_at=None,
        active_month_count=months,
        calendar_span_months=months,
        issues_by_month=months_map,
        labelled_issues_by_month=months_map,
    )
    return AuditAnalysis(
        repository_summary=summary,
        text_metrics=text,
        label_metrics=labels,
        temporal_metrics=temporal,
    )


def _codes(analysis: AuditAnalysis) -> set[str]:
    return {warning.code for warning in build_warnings(analysis)}


def test_healthy_dataset_has_no_warnings() -> None:
    assert build_warnings(_analysis()) == []


def test_insufficient_labelled_issues_boundary() -> None:
    assert "INSUFFICIENT_LABELLED_ISSUES" in _codes(_analysis(total=1000, labelled=499))
    assert "INSUFFICIENT_LABELLED_ISSUES" not in _codes(_analysis(total=1000, labelled=500))


def test_high_unlabelled_rate_boundary() -> None:
    assert "HIGH_UNLABELLED_RATE" in _codes(_analysis(total=1000, labelled=700))
    assert "HIGH_UNLABELLED_RATE" not in _codes(_analysis(total=1000, labelled=800))


def test_limited_temporal_coverage_boundary() -> None:
    assert "LIMITED_TEMPORAL_COVERAGE" in _codes(_analysis(months=5))
    assert "LIMITED_TEMPORAL_COVERAGE" not in _codes(_analysis(months=6))


def test_severe_label_long_tail_boundary() -> None:
    assert "SEVERE_LABEL_LONG_TAIL" in _codes(_analysis(unique_labels=10, lt_10=6))
    assert "SEVERE_LABEL_LONG_TAIL" not in _codes(_analysis(unique_labels=10, lt_10=5))


def test_low_text_completeness_boundary() -> None:
    assert "LOW_TEXT_COMPLETENESS" in _codes(_analysis(total=1000, short_count=310))
    assert "LOW_TEXT_COMPLETENESS" not in _codes(_analysis(total=1000, short_count=300))


def test_warnings_sorted_by_code() -> None:
    analysis = _analysis(
        total=1000,
        labelled=100,
        months=2,
        unique_labels=10,
        lt_10=9,
        short_count=900,
    )
    codes = [warning.code for warning in build_warnings(analysis)]
    assert codes == sorted(codes)
    assert len(codes) >= 4


def test_warning_fields_are_populated() -> None:
    warnings = build_warnings(_analysis(total=1000, labelled=499))
    warning = next(w for w in warnings if w.code == "INSUFFICIENT_LABELLED_ISSUES")
    assert warning.severity == "high"
    assert warning.value == 499.0
    assert warning.threshold == 500.0
    assert warning.message
