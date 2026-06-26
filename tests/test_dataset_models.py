"""Tests for dataset Pydantic models, validated types, and invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from repotriage.dataset.models import (
    NORMALIZER_VERSION,
    NormalizedIssue,
    ProcessedManifest,
    compute_dataset_id,
)

SOURCE_FETCHED_AT = datetime(2026, 6, 24, 16, 29, 50, 93080, tzinfo=UTC)
SOURCE_SNAPSHOT_SHA256 = "a" * 64
SOURCE_MANIFEST_SHA256 = "b" * 64
OUTPUT_SHA256 = "c" * 64
CONSISTENT_DATASET_ID = compute_dataset_id(
    SOURCE_FETCHED_AT, NORMALIZER_VERSION, SOURCE_SNAPSHOT_SHA256
)


def _base_manifest_kwargs() -> dict:
    return {
        "dataset_id": CONSISTENT_DATASET_ID,
        "repository": "o/r",
        "normalizer_version": NORMALIZER_VERSION,
        "built_at": datetime.now(UTC),
        "source_manifest": "o__r/manifest.json",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
        "source_manifest_schema_version": "2",
        "source_fetched_at": SOURCE_FETCHED_AT,
        "source_api_version": "2026-03-10",
        "source_pages_fetched": 2,
        "raw_records_read": 3,
        "pull_requests_excluded": 1,
        "issues_written": 2,
        "unlabelled_issues": 1,
        "empty_body_issues": 1,
        "output_file": "issues.jsonl",
        "output_sha256": OUTPUT_SHA256,
    }


def test_processed_manifest_accepts_valid_counts() -> None:
    manifest = ProcessedManifest(**_base_manifest_kwargs())
    assert manifest.schema_version == "1"
    assert manifest.issue_schema_version == "1"
    assert manifest.source_manifest_schema_version == "2"
    assert manifest.raw_records_read == 3


def test_processed_manifest_rejects_inconsistent_recon() -> None:
    with pytest.raises(ValidationError, match="raw_records_read"):
        ProcessedManifest(**{**_base_manifest_kwargs(), "issues_written": 5})


def test_processed_manifest_rejects_unlabelled_over_written() -> None:
    kwargs = _base_manifest_kwargs()
    kwargs.update({"raw_records_read": 2, "pull_requests_excluded": 0, "issues_written": 2})
    kwargs["unlabelled_issues"] = 3
    with pytest.raises(ValidationError, match="unlabelled_issues"):
        ProcessedManifest(**kwargs)


def test_processed_manifest_rejects_empty_body_over_written() -> None:
    kwargs = _base_manifest_kwargs()
    kwargs.update({"raw_records_read": 2, "pull_requests_excluded": 0, "issues_written": 2})
    kwargs["empty_body_issues"] = 3
    with pytest.raises(ValidationError, match="empty_body_issues"):
        ProcessedManifest(**kwargs)


def test_processed_manifest_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        ProcessedManifest(**{**_base_manifest_kwargs(), "issues_written": -1})


@pytest.mark.parametrize("unsafe", ["/etc/passwd", "../escape.jsonl", "../../x.jsonl"])
def test_processed_manifest_rejects_unsafe_output_file(unsafe: str) -> None:
    with pytest.raises(ValidationError, match="output_file"):
        ProcessedManifest(**{**_base_manifest_kwargs(), "output_file": unsafe})


@pytest.mark.parametrize("unsafe", ["/abs/manifest.json", "../o__r/manifest.json"])
def test_processed_manifest_rejects_unsafe_source_manifest(unsafe: str) -> None:
    with pytest.raises(ValidationError, match="source_manifest"):
        ProcessedManifest(**{**_base_manifest_kwargs(), "source_manifest": unsafe})


@pytest.mark.parametrize(
    "field",
    ["source_manifest_sha256", "source_snapshot_sha256", "output_sha256"],
)
@pytest.mark.parametrize(
    "bad_hash",
    ["A" * 64, "a" * 63, "a" * 65, "g" * 64, " " + "a" * 64, "a" * 64 + " ", "xyz"],
)
def test_processed_manifest_rejects_bad_sha256(field: str, bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        ProcessedManifest(**{**_base_manifest_kwargs(), field: bad_hash})


@pytest.mark.parametrize(
    "bad_id",
    [
        "20260624T162950093080Z-n1",  # legacy: missing hash suffix
        "20260624T162950093080Z-n1-AAAAAAAAAAAA",  # uppercase hex
        "20260624T162950093080Z-n1-abc",  # too short suffix
        "20260624T162950Z-n1-a31f820c4d2e",  # missing microseconds
        "20260624T162950093080Z-n0-a31f820c4d2e",  # zero version
        " 20260624T162950093080Z-n1-a31f820c4d2e",  # leading whitespace
    ],
)
def test_processed_manifest_rejects_bad_dataset_id_format(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        ProcessedManifest(**{**_base_manifest_kwargs(), "dataset_id": bad_id})


def test_processed_manifest_rejects_inconsistent_dataset_id() -> None:
    other = compute_dataset_id(SOURCE_FETCHED_AT, NORMALIZER_VERSION, "f" * 64)
    with pytest.raises(ValidationError, match="dataset_id is inconsistent"):
        ProcessedManifest(**{**_base_manifest_kwargs(), "dataset_id": other})


def test_dataset_id_is_content_aware() -> None:
    base = compute_dataset_id(SOURCE_FETCHED_AT, "1", "a" * 64)
    assert base == compute_dataset_id(SOURCE_FETCHED_AT, "1", "a" * 64)
    assert base != compute_dataset_id(SOURCE_FETCHED_AT, "1", "b" * 64)
    assert base != compute_dataset_id(SOURCE_FETCHED_AT, "2", "a" * 64)
    other_time = datetime(2026, 6, 25, 1, 2, 3, 4, tzinfo=UTC)
    assert base != compute_dataset_id(other_time, "1", "a" * 64)


def _base_issue_kwargs() -> dict:
    return {
        "repository": "o/r",
        "issue_id": 10,
        "issue_number": 5,
        "title": "A title",
        "body": "",
        "labels": [],
        "state": "open",
        "created_at": datetime(2026, 6, 24, 16, 9, 3, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 24, 16, 13, 42, tzinfo=UTC),
        "comments_count": 0,
        "html_url": "https://github.com/o/r/issues/5",
        "source_page": "pages/page_0001.json",
    }


def test_normalized_issue_rejects_blank_title() -> None:
    with pytest.raises(ValidationError, match="title"):
        NormalizedIssue(**{**_base_issue_kwargs(), "title": "   "})


def test_normalized_issue_rejects_non_positive_ids() -> None:
    with pytest.raises(ValidationError):
        NormalizedIssue(**{**_base_issue_kwargs(), "issue_id": 0})
    with pytest.raises(ValidationError):
        NormalizedIssue(**{**_base_issue_kwargs(), "issue_number": 0})


def test_normalized_issue_rejects_unknown_state() -> None:
    with pytest.raises(ValidationError):
        NormalizedIssue(**{**_base_issue_kwargs(), "state": "merged"})


def test_normalized_issue_accepts_valid_github_url() -> None:
    issue = NormalizedIssue(
        **{
            **_base_issue_kwargs(),
            "repository": "pandas-dev/pandas",
            "issue_number": 66012,
            "html_url": "https://github.com/pandas-dev/pandas/issues/66012",
        }
    )
    assert issue.html_url.endswith("/issues/66012")


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://github.com/o/r/issues/5",
        "https://example.com/o/r/issues/5",
        "https://github.com.evil.com/o/r/issues/5",
        "https://user:pass@github.com/o/r/issues/5",
        "https://github.com/o/r/pull/5",
        "https://github.com/o/r",
    ],
)
def test_normalized_issue_rejects_bad_url(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        NormalizedIssue(**{**_base_issue_kwargs(), "html_url": bad_url})


def test_html_url_matching_repo_and_number_succeeds() -> None:
    issue = NormalizedIssue(
        **{
            **_base_issue_kwargs(),
            "repository": "pandas-dev/pandas",
            "issue_number": 123,
            "html_url": "https://github.com/pandas-dev/pandas/issues/123",
        }
    )
    assert issue.issue_number == 123


def test_html_url_owner_repo_compared_case_insensitively() -> None:
    issue = NormalizedIssue(
        **{
            **_base_issue_kwargs(),
            "repository": "Pandas-Dev/Pandas",
            "issue_number": 123,
            "html_url": "https://github.com/pandas-dev/pandas/issues/123",
        }
    )
    assert issue.html_url.endswith("/pandas/issues/123")


def test_html_url_wrong_owner_fails() -> None:
    with pytest.raises(ValidationError, match="owner"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": "https://github.com/scikit-learn/pandas/issues/123",
            }
        )


def test_html_url_wrong_repository_fails() -> None:
    with pytest.raises(ValidationError, match="repository"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": "https://github.com/pandas-dev/numpy/issues/123",
            }
        )


def test_html_url_wrong_issue_number_fails() -> None:
    with pytest.raises(ValidationError, match="issue number"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": "https://github.com/pandas-dev/pandas/issues/999",
            }
        )


def test_html_url_pull_request_path_fails() -> None:
    with pytest.raises(ValidationError, match="issue path"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": "https://github.com/pandas-dev/pandas/pull/123",
            }
        )


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://github.com/pandas-dev/pandas/issues/123/",  # trailing slash rejected
        "https://github.com/pandas-dev/pandas/issues/123/comments",  # extra component
        "https://github.com/pandas-dev/pandas/issues/123/files",
    ],
)
def test_html_url_extra_path_components_fail(bad_url: str) -> None:
    with pytest.raises(ValidationError, match="issue path"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": bad_url,
            }
        )


def test_html_url_query_string_fails() -> None:
    with pytest.raises(ValidationError, match="query string"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": "https://github.com/pandas-dev/pandas/issues/123?tab=foo",
            }
        )


def test_html_url_fragment_fails() -> None:
    with pytest.raises(ValidationError, match="fragment"):
        NormalizedIssue(
            **{
                **_base_issue_kwargs(),
                "repository": "pandas-dev/pandas",
                "issue_number": 123,
                "html_url": "https://github.com/pandas-dev/pandas/issues/123#note",
            }
        )
