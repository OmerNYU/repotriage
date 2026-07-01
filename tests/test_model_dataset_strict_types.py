"""Strict JSON integer typing tests for model-ready artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from repotriage.model_dataset.models import (
    GlobalTargetStatistics,
    LabelMap,
    ModelReadyRecord,
    SplitStatistics,
    SplitWarning,
)

_POLICY_ID = "20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37"
_BASE_RECORD = dict(
    schema_version="1",
    repository="pandas-dev/pandas",
    issue_number=1,
    created_at=datetime(2026, 1, 1, tzinfo=UTC),
    title="t",
    body="b",
    feature_text="ft",
    selected_labels=["Bug"],
    split="train",
)


@pytest.mark.parametrize(
    "vector",
    [[True], [False], [1.0], [0.0], ["1"], ["0"]],
)
def test_target_vector_rejects_coerced_values(vector: list[object]) -> None:
    with pytest.raises(ValidationError):
        ModelReadyRecord(**_BASE_RECORD, issue_id=1, target_vector=vector)


@pytest.mark.parametrize("issue_id", [True, 1.0, "1"])
def test_issue_id_rejects_coerced_values(issue_id: object) -> None:
    with pytest.raises(ValidationError):
        ModelReadyRecord(**_BASE_RECORD, issue_id=issue_id, target_vector=[1])


def test_target_vector_accepts_json_integers() -> None:
    record = ModelReadyRecord(**_BASE_RECORD, issue_id=1, target_vector=[1, 0])
    assert record.target_vector == [1, 0]


def test_label_map_rejects_boolean_indices() -> None:
    with pytest.raises(ValidationError):
        LabelMap(
            policy_id=_POLICY_ID,
            target_count=1,
            labels=["Bug"],
            label_to_index={"Bug": True},
        )


def test_split_warning_rejects_float_count() -> None:
    with pytest.raises(ValidationError):
        SplitWarning(label="Bug", split="test", count=1.0, threshold=5)


def test_split_statistics_rejects_float_issue_count() -> None:
    with pytest.raises(ValidationError):
        SplitStatistics(
            issue_count=1.0,
            fraction=1.0,
            all_zero_target_count=0,
        )


def test_global_statistics_reject_float_counts() -> None:
    with pytest.raises(ValidationError):
        GlobalTargetStatistics(
            total_records=1.0,
            target_count=1,
            issues_with_included_target=1,
            issues_without_included_target=0,
            target_coverage_fraction=1.0,
            positive_assignments=1,
            all_zero_target_count=0,
        )


def test_json_parse_rejects_boolean_target_vector() -> None:
    payload = {
        **_BASE_RECORD,
        "issue_id": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "target_vector": [True],
    }
    with pytest.raises(ValidationError):
        ModelReadyRecord.model_validate(json.loads(json.dumps(payload)))
