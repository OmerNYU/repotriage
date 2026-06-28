"""Tests for objective audit statistics."""

from __future__ import annotations

from datetime import UTC, datetime

from repotriage.audit.analyzer import Analyzer, _percentile, _text_field_stats
from tests.helpers import make_normalized_issue


def _at(year: int, month: int, day: int = 1) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def _analyze(issues: list) -> object:
    analyzer = Analyzer()
    for issue in issues:
        analyzer.add(issue)
    return analyzer.finalize()


def test_repository_summary_counts_and_fractions() -> None:
    issues = [
        make_normalized_issue(1, labels=["Bug"]),
        make_normalized_issue(2, labels=["Bug", "Docs"]),
        make_normalized_issue(3, labels=[]),
        make_normalized_issue(4, labels=[]),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.total_issues == 4
    assert summary.labelled_issues == 2
    assert summary.unlabelled_issues == 2
    assert summary.labelled_fraction == 0.5
    assert summary.unlabelled_fraction == 0.5


def test_state_counts() -> None:
    issues = [
        make_normalized_issue(1, state="open"),
        make_normalized_issue(2, state="closed", closed_at=_at(2025, 4)),
        make_normalized_issue(3, state="closed", closed_at=_at(2025, 4)),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.open_issues == 1
    assert summary.closed_issues == 2


def test_null_author_count() -> None:
    issues = [
        make_normalized_issue(1, author_login="octocat"),
        make_normalized_issue(2, author_login=None),
        make_normalized_issue(3, author_login=None),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.null_author_issues == 2


def test_label_frequencies_and_fractions() -> None:
    issues = [
        make_normalized_issue(1, labels=["Bug"]),
        make_normalized_issue(2, labels=["Bug", "Docs"]),
        make_normalized_issue(3, labels=["Docs"]),
        make_normalized_issue(4, labels=[]),
    ]
    labels = _analyze(issues).label_metrics
    by_name = {label.name: label for label in labels.labels}
    assert by_name["Bug"].count == 2
    assert by_name["Docs"].count == 2
    assert by_name["Bug"].fraction == 0.5
    # Ties on count fall back to ascending name order.
    assert [label.name for label in labels.labels] == ["Bug", "Docs"]


def test_label_first_and_last_occurrence() -> None:
    issues = [
        make_normalized_issue(1, labels=["Bug"], created_at=_at(2025, 3)),
        make_normalized_issue(2, labels=["Bug"], created_at=_at(2025, 6)),
        make_normalized_issue(3, labels=["Bug"], created_at=_at(2025, 1)),
    ]
    labels = _analyze(issues).label_metrics
    bug = labels.labels[0]
    assert bug.first_created_at == _at(2025, 1)
    assert bug.last_created_at == _at(2025, 6)


def test_label_cardinality_and_density() -> None:
    issues = [
        make_normalized_issue(1, labels=["Bug", "Docs"]),
        make_normalized_issue(2, labels=["Bug"]),
    ]
    labels = _analyze(issues).label_metrics
    assert labels.total_label_assignments == 3
    assert labels.unique_label_count == 2
    assert labels.label_cardinality == 3 / 2
    assert labels.label_density == (3 / 2) / 2


def test_rare_label_buckets_use_strict_thresholds() -> None:
    # Label "Five" appears in exactly 5 issues: excluded from lt_5, included in lt_10.
    issues = []
    number = 1
    for _ in range(5):
        issues.append(make_normalized_issue(number, labels=["Five"]))
        number += 1
    for _ in range(3):
        issues.append(make_normalized_issue(number, labels=["Three"]))
        number += 1
    buckets = _analyze(issues).label_metrics.rare_label_buckets
    assert buckets.lt_5 == 1  # only "Three"
    assert buckets.lt_10 == 2  # "Three" and "Five"


def test_label_pair_canonicalization_and_order() -> None:
    issues = [
        make_normalized_issue(1, labels=["Docs", "Bug"]),
        make_normalized_issue(2, labels=["Bug", "Docs"]),
    ]
    pairs = _analyze(issues).label_metrics.label_pairs
    assert len(pairs) == 1
    assert (pairs[0].label_a, pairs[0].label_b) == ("Bug", "Docs")
    assert pairs[0].count == 2


def test_label_pair_minimum_support() -> None:
    issues = [
        make_normalized_issue(1, labels=["Bug", "Docs"]),
        make_normalized_issue(2, labels=["Bug", "Docs"]),
        make_normalized_issue(3, labels=["Bug", "Perf"]),
    ]
    pairs = _analyze(issues).label_metrics.label_pairs
    keys = {(pair.label_a, pair.label_b) for pair in pairs}
    assert ("Bug", "Docs") in keys  # co-occurs twice
    assert ("Bug", "Perf") not in keys  # co-occurs once


def test_percentile_exact_interpolation() -> None:
    sample = [0, 1, 2, 3, 4]
    assert _percentile(sample, 50) == 2.0
    assert _percentile(sample, 90) == 3.6
    assert _percentile(sample, 95) == 3.8
    # Even-length median interpolates between the two central values.
    assert _percentile([1, 2], 50) == 1.5


def test_text_field_stats_empty_population_is_none() -> None:
    stats = _text_field_stats([])
    assert stats.min is None
    assert stats.median is None
    assert stats.max is None


def test_text_structural_indicators_and_boundaries() -> None:
    issues = [
        make_normalized_issue(1, body=""),  # empty + short
        make_normalized_issue(2, body="x" * 99),  # short (<100)
        make_normalized_issue(3, body="x" * 100),  # not short (==100)
        make_normalized_issue(4, body="```python\ncode\n```"),  # code fence
        make_normalized_issue(5, body="see https://example.com here"),  # url
        make_normalized_issue(6, body="# Heading\nbody"),  # heading
        make_normalized_issue(7, body="x" * 10001),  # long (>10000)
        make_normalized_issue(8, body="x" * 10000),  # not long (==10000)
    ]
    structural = _analyze(issues).text_metrics.structural
    assert structural.empty_bodies.count == 1
    # Issues 1, 2, 4, 5, 6 all have bodies shorter than 100 chars; issue 3 (==100) is not.
    assert structural.short_bodies_lt_100.count == 5
    # Only issue 7 (>10000) counts; issue 8 (==10000) does not.
    assert structural.long_bodies_gt_10000.count == 1
    assert structural.with_code_fence.count == 1
    assert structural.with_url.count == 1
    assert structural.with_heading.count == 1
    assert structural.empty_bodies.fraction == 1 / 8


def test_monthly_temporal_counts() -> None:
    issues = [
        make_normalized_issue(1, labels=["Bug"], created_at=_at(2025, 1, 5)),
        make_normalized_issue(2, labels=[], created_at=_at(2025, 1, 20)),
        make_normalized_issue(3, labels=["Bug"], created_at=_at(2025, 3, 2)),
    ]
    analysis = _analyze(issues)
    temporal = analysis.temporal_metrics
    assert temporal.active_month_count == 2
    assert temporal.calendar_span_months == 3
    assert analysis.repository_summary.active_month_count == 2
    assert analysis.repository_summary.calendar_span_months == 3
    assert list(temporal.issues_by_month.keys()) == ["2025-01", "2025-03"]
    assert temporal.issues_by_month == {"2025-01": 2, "2025-03": 1}
    assert temporal.labelled_issues_by_month == {"2025-01": 1, "2025-03": 1}


def test_calendar_span_months_sparse_long_span() -> None:
    issues = [
        make_normalized_issue(1, created_at=_at(2022, 2, 15)),
        make_normalized_issue(2, created_at=_at(2025, 4, 10)),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.active_month_count == 2
    assert summary.calendar_span_months == (2025 - 2022) * 12 + 4 - 2 + 1


def test_calendar_span_months_contiguous() -> None:
    issues = [
        make_normalized_issue(1, created_at=_at(2025, 1, 5)),
        make_normalized_issue(2, created_at=_at(2025, 2, 5)),
        make_normalized_issue(3, created_at=_at(2025, 3, 5)),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.active_month_count == 3
    assert summary.calendar_span_months == 3


def test_calendar_span_months_same_month() -> None:
    issues = [
        make_normalized_issue(1, created_at=_at(2025, 6, 1)),
        make_normalized_issue(2, created_at=_at(2025, 6, 28)),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.active_month_count == 1
    assert summary.calendar_span_months == 1


def test_total_text_chars() -> None:
    issues = [
        make_normalized_issue(1, title="abcde", body="xyz"),
    ]
    text = _analyze(issues).text_metrics
    assert text.total_text_chars.max == 8


def test_temporal_span_days() -> None:
    issues = [
        make_normalized_issue(1, created_at=_at(2025, 1, 1)),
        make_normalized_issue(2, created_at=_at(2025, 1, 11)),
    ]
    summary = _analyze(issues).repository_summary
    assert summary.temporal_span_days == 10.0
