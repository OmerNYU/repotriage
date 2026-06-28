"""Tests for tolerant float reconciliation in audit models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from repotriage.audit.models import (
    LabelFrequency,
    LabelMetrics,
    LabelsPerIssueStats,
    RareLabelBuckets,
    RepositorySummary,
)


def _summary(*, labelled_fraction: float, unlabelled_fraction: float) -> RepositorySummary:
    return RepositorySummary(
        total_issues=1000,
        labelled_issues=600,
        unlabelled_issues=400,
        labelled_fraction=labelled_fraction,
        unlabelled_fraction=unlabelled_fraction,
        open_issues=1000,
        closed_issues=0,
        null_author_issues=0,
        earliest_created_at=None,
        latest_created_at=None,
        temporal_span_days=0.0,
        active_month_count=0,
        calendar_span_months=0,
    )


def test_repository_summary_accepts_tiny_float_perturbation() -> None:
    summary = _summary(labelled_fraction=0.6 + 1e-15, unlabelled_fraction=0.4 - 1e-15)
    assert summary.labelled_issues == 600


def test_repository_summary_rejects_material_fraction_error() -> None:
    with pytest.raises(ValidationError, match="labelled_fraction"):
        _summary(labelled_fraction=0.61, unlabelled_fraction=0.4)


def _labels(*, label_density: float) -> LabelMetrics:
    moment = datetime(2025, 3, 1, 12, 0, 0, tzinfo=UTC)
    labels = [
        LabelFrequency(
            name=f"label-{i}",
            count=1,
            fraction=0.0,
            first_created_at=moment,
            last_created_at=moment,
        )
        for i in range(10)
    ]
    return LabelMetrics(
        unique_label_count=10,
        total_label_assignments=2,
        zero_label_issue_count=0,
        labels_per_issue=LabelsPerIssueStats(min=0, median=0.0, mean=0.0, max=0),
        label_cardinality=2.0,
        label_density=label_density,
        rare_label_buckets=RareLabelBuckets(lt_5=0, lt_10=0, lt_20=0, lt_50=0, lt_100=0),
        labels=labels,
        label_pairs=[],
    )


def test_label_metrics_accepts_tiny_float_perturbation() -> None:
    labels = _labels(label_density=0.2 + 1e-15)
    assert labels.unique_label_count == 10


def test_label_metrics_rejects_material_density_error() -> None:
    with pytest.raises(ValidationError, match="label_density"):
        _labels(label_density=0.25)
