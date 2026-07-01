"""Tests for deterministic model-dataset report serialization."""

from datetime import UTC, datetime

from repotriage.model_dataset.models import (
    GlobalTargetStatistics,
    SplitReport,
    SplitStatistics,
    SupportValidationSummary,
)
from repotriage.model_dataset.report import (
    serialize_split_report_json,
    serialize_split_report_markdown,
    sha256_hex,
)


def _sample_report() -> SplitReport:
    return SplitReport(
        split_strategy="temporal_calendar",
        validation_start=datetime(2026, 2, 1, tzinfo=UTC),
        test_start=datetime(2026, 4, 1, tzinfo=UTC),
        boundary_semantics={
            "train": "created_at < validation_start",
            "validation": "validation_start <= created_at < test_start",
            "test": "created_at >= test_start",
        },
        total_records=10,
        global_target_statistics=GlobalTargetStatistics(
            total_records=10,
            target_count=2,
            issues_with_included_target=8,
            issues_without_included_target=2,
            target_coverage_fraction=0.8,
            positive_assignments=9,
            all_zero_target_count=2,
        ),
        splits={
            "train": SplitStatistics(
                issue_count=6,
                fraction=0.6,
                earliest_created_at=datetime(2025, 1, 1, tzinfo=UTC),
                latest_created_at=datetime(2026, 1, 1, tzinfo=UTC),
                all_zero_target_count=1,
                target_cardinality_histogram={"0": 1, "1": 5},
                positives_per_label={"Bug": 4, "Docs": 2},
            ),
            "validation": SplitStatistics(
                issue_count=2,
                fraction=0.2,
                earliest_created_at=datetime(2026, 2, 1, tzinfo=UTC),
                latest_created_at=datetime(2026, 3, 1, tzinfo=UTC),
                all_zero_target_count=0,
                target_cardinality_histogram={"1": 2},
                positives_per_label={"Bug": 2, "Docs": 1},
            ),
            "test": SplitStatistics(
                issue_count=2,
                fraction=0.2,
                earliest_created_at=datetime(2026, 4, 1, tzinfo=UTC),
                latest_created_at=datetime(2026, 5, 1, tzinfo=UTC),
                all_zero_target_count=1,
                target_cardinality_histogram={"0": 1, "1": 1},
                positives_per_label={"Bug": 1, "Docs": 0},
            ),
        },
        warnings=[],
        support_validation=SupportValidationSummary(hard_errors=[], warnings=[]),
    )


def test_deterministic_json_and_markdown() -> None:
    report = _sample_report()
    json_a = serialize_split_report_json(report)
    json_b = serialize_split_report_json(report)
    md_a = serialize_split_report_markdown(report)
    md_b = serialize_split_report_markdown(report)
    assert json_a == json_b
    assert md_a == md_b
    assert sha256_hex(json_a) == sha256_hex(json_b)
    assert b"built_at" not in json_a
