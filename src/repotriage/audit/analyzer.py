"""Objective statistics over normalized issues.

This module is policy-free: it computes deterministic, objective metrics only. The
:class:`Analyzer` is a single-pass accumulator that retains compact derived values
(lengths, counters, per-label and per-pair tallies) rather than full issue bodies.

Percentiles and the median use a single, explicitly defined rule (type-7 linear
interpolation) so results never depend on an unspecified library default.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from repotriage.audit.models import (
    CountFraction,
    LabelFrequency,
    LabelMetrics,
    LabelPair,
    LabelsPerIssueStats,
    RareLabelBuckets,
    RepositorySummary,
    TemporalMetrics,
    TextFieldStats,
    TextMetrics,
    TextStructuralIndicators,
)
from repotriage.dataset.models import NormalizedIssue

# Structural, deterministic detectors. These are intentionally simple structural
# indicators, not a Markdown parser or NLP cleaner.
_CODE_FENCE_RE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S")
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s", re.MULTILINE)

_RARE_LABEL_THRESHOLDS = (5, 10, 20, 50, 100)


def _percentile(sorted_values: list[int], percentile: float) -> float:
    """Type-7 linear-interpolation percentile over an ascending, non-empty sample.

    ``rank = (p/100) * (n - 1)``; the result interpolates linearly between the values
    bracketing that fractional zero-based rank. ``percentile(50)`` is the median.
    """
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


def _text_field_stats(values: list[int]) -> TextFieldStats:
    if not values:
        return TextFieldStats()
    ordered = sorted(values)
    return TextFieldStats(
        min=ordered[0],
        median=_percentile(ordered, 50),
        mean=sum(ordered) / len(ordered),
        p90=_percentile(ordered, 90),
        p95=_percentile(ordered, 95),
        max=ordered[-1],
    )


@dataclass
class _LabelTally:
    count: int = 0
    first_created_at: datetime | None = None
    last_created_at: datetime | None = None


@dataclass
class AuditAnalysis:
    """Objective metric sections produced by the analyzer."""

    repository_summary: RepositorySummary
    text_metrics: TextMetrics
    label_metrics: LabelMetrics
    temporal_metrics: TemporalMetrics


@dataclass
class Analyzer:
    """Single-pass accumulator of objective metrics over normalized issues."""

    total_issues: int = 0
    labelled_issues: int = 0
    open_issues: int = 0
    closed_issues: int = 0
    null_author_issues: int = 0
    earliest_created_at: datetime | None = None
    latest_created_at: datetime | None = None

    _title_lengths: list[int] = field(default_factory=list)
    _body_lengths: list[int] = field(default_factory=list)
    _total_text_lengths: list[int] = field(default_factory=list)

    _empty_bodies: int = 0
    _short_bodies: int = 0
    _long_bodies: int = 0
    _with_code_fence: int = 0
    _with_url: int = 0
    _with_heading: int = 0

    _labels_per_issue: list[int] = field(default_factory=list)
    _total_label_assignments: int = 0
    _label_tallies: dict[str, _LabelTally] = field(default_factory=dict)
    _pair_counts: Counter[tuple[str, str]] = field(default_factory=Counter)

    _issues_by_month: Counter[str] = field(default_factory=Counter)
    _labelled_by_month: Counter[str] = field(default_factory=Counter)

    def add(self, issue: NormalizedIssue) -> None:
        self.total_issues += 1

        if issue.state == "open":
            self.open_issues += 1
        else:
            self.closed_issues += 1

        if issue.author_login is None:
            self.null_author_issues += 1

        created = issue.created_at
        if self.earliest_created_at is None or created < self.earliest_created_at:
            self.earliest_created_at = created
        if self.latest_created_at is None or created > self.latest_created_at:
            self.latest_created_at = created

        title_chars = len(issue.title)
        body_chars = len(issue.body)
        self._title_lengths.append(title_chars)
        self._body_lengths.append(body_chars)
        self._total_text_lengths.append(title_chars + body_chars)

        if issue.body == "":
            self._empty_bodies += 1
        if body_chars < 100:
            self._short_bodies += 1
        if body_chars > 10000:
            self._long_bodies += 1
        if _CODE_FENCE_RE.search(issue.body) is not None:
            self._with_code_fence += 1
        if _URL_RE.search(issue.body) is not None:
            self._with_url += 1
        if _HEADING_RE.search(issue.body) is not None:
            self._with_heading += 1

        labels = sorted(set(issue.labels))
        self._labels_per_issue.append(len(labels))
        self._total_label_assignments += len(labels)
        if labels:
            self.labelled_issues += 1

        for label in labels:
            tally = self._label_tallies.get(label)
            if tally is None:
                tally = _LabelTally()
                self._label_tallies[label] = tally
            tally.count += 1
            if tally.first_created_at is None or created < tally.first_created_at:
                tally.first_created_at = created
            if tally.last_created_at is None or created > tally.last_created_at:
                tally.last_created_at = created

        for index, label_a in enumerate(labels):
            for label_b in labels[index + 1 :]:
                self._pair_counts[(label_a, label_b)] += 1

        month_key = created.strftime("%Y-%m")
        self._issues_by_month[month_key] += 1
        self._labelled_by_month[month_key] += 1 if labels else 0

    def _build_repository_summary(self) -> RepositorySummary:
        total = self.total_issues
        unlabelled = total - self.labelled_issues
        labelled_fraction = self.labelled_issues / total if total else 0.0
        unlabelled_fraction = unlabelled / total if total else 0.0
        if self.earliest_created_at is not None and self.latest_created_at is not None:
            span_seconds = (self.latest_created_at - self.earliest_created_at).total_seconds()
            span_days = span_seconds / 86400.0
        else:
            span_days = 0.0
        return RepositorySummary(
            total_issues=total,
            labelled_issues=self.labelled_issues,
            unlabelled_issues=unlabelled,
            labelled_fraction=labelled_fraction,
            unlabelled_fraction=unlabelled_fraction,
            open_issues=self.open_issues,
            closed_issues=self.closed_issues,
            null_author_issues=self.null_author_issues,
            earliest_created_at=self.earliest_created_at,
            latest_created_at=self.latest_created_at,
            temporal_span_days=span_days,
            active_month_count=len(self._issues_by_month),
            calendar_span_months=self._calendar_span_months(),
        )

    def _build_text_metrics(self) -> TextMetrics:
        total = self.total_issues

        def fraction(count: int) -> float:
            return count / total if total else 0.0

        structural = TextStructuralIndicators(
            empty_bodies=CountFraction(
                count=self._empty_bodies, fraction=fraction(self._empty_bodies)
            ),
            short_bodies_lt_100=CountFraction(
                count=self._short_bodies, fraction=fraction(self._short_bodies)
            ),
            long_bodies_gt_10000=CountFraction(
                count=self._long_bodies, fraction=fraction(self._long_bodies)
            ),
            with_code_fence=CountFraction(
                count=self._with_code_fence, fraction=fraction(self._with_code_fence)
            ),
            with_url=CountFraction(count=self._with_url, fraction=fraction(self._with_url)),
            with_heading=CountFraction(
                count=self._with_heading, fraction=fraction(self._with_heading)
            ),
        )
        return TextMetrics(
            title_chars=_text_field_stats(self._title_lengths),
            body_chars=_text_field_stats(self._body_lengths),
            total_text_chars=_text_field_stats(self._total_text_lengths),
            structural=structural,
        )

    def _build_label_metrics(self) -> LabelMetrics:
        total = self.total_issues
        unique = len(self._label_tallies)

        if self._labels_per_issue:
            ordered = sorted(self._labels_per_issue)
            labels_per_issue = LabelsPerIssueStats(
                min=ordered[0],
                median=_percentile(ordered, 50),
                mean=sum(ordered) / len(ordered),
                max=ordered[-1],
            )
        else:
            labels_per_issue = LabelsPerIssueStats(min=0, median=0.0, mean=0.0, max=0)

        cardinality = self._total_label_assignments / total if total else 0.0
        density = cardinality / unique if unique else 0.0

        buckets = {threshold: 0 for threshold in _RARE_LABEL_THRESHOLDS}
        for tally in self._label_tallies.values():
            for threshold in _RARE_LABEL_THRESHOLDS:
                if tally.count < threshold:
                    buckets[threshold] += 1

        labels = [
            LabelFrequency(
                name=name,
                count=tally.count,
                fraction=tally.count / total if total else 0.0,
                first_created_at=tally.first_created_at,
                last_created_at=tally.last_created_at,
            )
            for name, tally in self._label_tallies.items()
        ]
        labels.sort(key=lambda item: (-item.count, item.name))

        pairs = [
            LabelPair(label_a=pair[0], label_b=pair[1], count=count)
            for pair, count in self._pair_counts.items()
            if count >= 2
        ]
        pairs.sort(key=lambda item: (-item.count, item.label_a, item.label_b))

        return LabelMetrics(
            unique_label_count=unique,
            total_label_assignments=self._total_label_assignments,
            zero_label_issue_count=total - self.labelled_issues,
            labels_per_issue=labels_per_issue,
            label_cardinality=cardinality,
            label_density=density,
            rare_label_buckets=RareLabelBuckets(
                lt_5=buckets[5],
                lt_10=buckets[10],
                lt_20=buckets[20],
                lt_50=buckets[50],
                lt_100=buckets[100],
            ),
            labels=labels,
            label_pairs=pairs,
        )

    def _calendar_span_months(self) -> int:
        """Inclusive count of calendar months between the earliest and latest issue.

        Distinct from ``active_month_count`` (months that actually contain issues): a
        dataset spanning Jan and Dec of one year has a calendar span of 12 even if only
        two of those months are active.
        """
        earliest = self.earliest_created_at
        latest = self.latest_created_at
        if earliest is None or latest is None:
            return 0
        return (latest.year - earliest.year) * 12 + latest.month - earliest.month + 1

    def _build_temporal_metrics(self) -> TemporalMetrics:
        months = sorted(self._issues_by_month)
        issues_by_month = {month: self._issues_by_month[month] for month in months}
        labelled_by_month = {month: self._labelled_by_month[month] for month in months}
        return TemporalMetrics(
            earliest_created_at=self.earliest_created_at,
            latest_created_at=self.latest_created_at,
            active_month_count=len(months),
            calendar_span_months=self._calendar_span_months(),
            issues_by_month=issues_by_month,
            labelled_issues_by_month=labelled_by_month,
        )

    def finalize(self) -> AuditAnalysis:
        return AuditAnalysis(
            repository_summary=self._build_repository_summary(),
            text_metrics=self._build_text_metrics(),
            label_metrics=self._build_label_metrics(),
            temporal_metrics=self._build_temporal_metrics(),
        )
