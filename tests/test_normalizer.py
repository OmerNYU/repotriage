"""Tests for the pure raw-to-normalized issue transformation."""

from __future__ import annotations

from datetime import UTC

import pytest

from repotriage.dataset.builder import serialize_issues_jsonl
from repotriage.dataset.models import MalformedIssueError, RawLabel
from repotriage.dataset.normalizer import normalize_issue, normalize_labels
from tests.helpers import make_raw_issue


def _normalize(raw: dict, *, repository: str = "pandas-dev/pandas", position: int = 0):
    return normalize_issue(
        raw,
        repository=repository,
        source_page="pages/page_0001.json",
        position=position,
    )


def test_null_body_becomes_empty_string() -> None:
    issue = _normalize(make_raw_issue(1, body=None))
    assert issue.body == ""


def test_present_body_is_preserved() -> None:
    issue = _normalize(make_raw_issue(1, body="hello world"))
    assert issue.body == "hello world"


def test_null_user_is_accepted() -> None:
    issue = _normalize(make_raw_issue(1, user=None))
    assert issue.author_login is None
    assert issue.author_type is None


def test_user_fields_are_extracted() -> None:
    issue = _normalize(make_raw_issue(1, user={"login": "octocat", "type": "User"}))
    assert issue.author_login == "octocat"
    assert issue.author_type == "User"


def test_label_extraction_dedup_and_deterministic_sort() -> None:
    issue = _normalize(
        make_raw_issue(1, labels=["zebra", "Bug", "apple", "Bug", "Apple"])
    )
    assert issue.labels == ["Apple", "Bug", "apple", "zebra"]


def test_empty_label_list_is_preserved() -> None:
    issue = _normalize(make_raw_issue(1, labels=[]))
    assert issue.labels == []


def test_normalize_labels_is_case_sensitive_codepoint_order() -> None:
    labels = [RawLabel(name="b"), RawLabel(name="A"), RawLabel(name="a"), RawLabel(name="A")]
    assert normalize_labels(labels) == ["A", "a", "b"]


def test_timestamps_normalized_to_utc() -> None:
    issue = _normalize(
        make_raw_issue(
            1,
            created_at="2026-06-24T21:09:03+05:00",
            updated_at="2026-06-24T16:13:42Z",
            closed_at="2026-06-25T00:00:00+02:00",
        )
    )
    assert issue.created_at.tzinfo == UTC
    assert issue.created_at.hour == 16
    assert issue.updated_at.tzinfo == UTC
    assert issue.closed_at is not None
    assert issue.closed_at.tzinfo == UTC
    assert issue.closed_at.hour == 22


def test_closed_at_null_is_allowed() -> None:
    issue = _normalize(make_raw_issue(1, closed_at=None))
    assert issue.closed_at is None


def test_malformed_issue_raises_with_source_context() -> None:
    raw = make_raw_issue(42)
    del raw["title"]
    with pytest.raises(MalformedIssueError) as exc_info:
        _normalize(raw, position=3)
    message = str(exc_info.value)
    assert "pages/page_0001.json" in message
    assert "position 3" in message
    assert "issue 42" in message


def test_blank_title_is_malformed() -> None:
    with pytest.raises(MalformedIssueError):
        _normalize(make_raw_issue(1, title="   "))


def test_non_positive_issue_number_is_malformed() -> None:
    with pytest.raises(MalformedIssueError):
        _normalize(make_raw_issue(0))


def test_offset_timestamp_serializes_as_utc_z() -> None:
    issue = _normalize(make_raw_issue(1, created_at="2026-06-24T21:09:03+05:00"))
    assert issue.model_dump(mode="json")["created_at"] == "2026-06-24T16:09:03Z"


def test_z_and_zero_offset_inputs_serialize_identically() -> None:
    z_issue = _normalize(make_raw_issue(1, created_at="2026-06-24T16:09:03Z"))
    offset_issue = _normalize(make_raw_issue(1, created_at="2026-06-24T16:09:03+00:00"))
    assert (
        z_issue.model_dump(mode="json")["created_at"]
        == offset_issue.model_dump(mode="json")["created_at"]
        == "2026-06-24T16:09:03Z"
    )


def test_microseconds_emitted_when_nonzero() -> None:
    issue = _normalize(make_raw_issue(1, created_at="2026-06-24T16:09:03.093080Z"))
    assert issue.model_dump(mode="json")["created_at"] == "2026-06-24T16:09:03.093080Z"


def test_null_closed_at_serializes_as_json_null() -> None:
    issue = _normalize(make_raw_issue(1, closed_at=None))
    assert issue.model_dump(mode="json")["closed_at"] is None


def test_repeated_serialization_is_byte_identical() -> None:
    issue = _normalize(make_raw_issue(1, created_at="2026-06-24T21:09:03+05:00"))
    first = serialize_issues_jsonl([issue])
    second = serialize_issues_jsonl([issue])
    assert first == second
    assert b"2026-06-24T16:09:03Z" in first
