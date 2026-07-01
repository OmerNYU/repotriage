"""Tests for the streaming normalized-issue JSONL reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.dataset.builder import serialize_issues_jsonl
from repotriage.dataset.models import DatasetReadError, ProcessedManifest
from repotriage.dataset.reader import iter_issues, read_dataset_issues
from repotriage.github.models import RepositoryRef
from tests.helpers import make_normalized_issue, write_processed_dataset


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_valid_jsonl_streaming(tmp_path: Path) -> None:
    issues = [make_normalized_issue(1), make_normalized_issue(2, labels=["Bug"])]
    path = tmp_path / "issues.jsonl"
    path.write_bytes(serialize_issues_jsonl(issues))

    read = list(iter_issues(path))
    assert [issue.issue_number for issue in read] == [1, 2]
    assert read[1].labels == ["Bug"]


def test_blank_line_rejected_with_context(tmp_path: Path) -> None:
    issues = [make_normalized_issue(1), make_normalized_issue(2)]
    serialized = serialize_issues_jsonl(issues).decode("utf-8").splitlines()
    path = tmp_path / "issues.jsonl"
    _write_lines(path, [serialized[0], "", serialized[1]])

    with pytest.raises(DatasetReadError) as exc_info:
        list(iter_issues(path))
    assert "Blank line" in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    good = serialize_issues_jsonl([make_normalized_issue(1)]).decode("utf-8").strip()
    path = tmp_path / "issues.jsonl"
    _write_lines(path, [good, "{not valid json"])

    with pytest.raises(DatasetReadError) as exc_info:
        list(iter_issues(path))
    assert "Malformed JSON" in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_invalid_normalized_issue_reports_line_number(tmp_path: Path) -> None:
    good = serialize_issues_jsonl([make_normalized_issue(1)]).decode("utf-8").strip()
    path = tmp_path / "issues.jsonl"
    _write_lines(path, [good, '{"issue_number": 2}'])

    with pytest.raises(DatasetReadError) as exc_info:
        list(iter_issues(path))
    assert "Invalid normalized issue" in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_record_count_mismatch_against_manifest(tmp_path: Path) -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    processed_root = tmp_path / "processed"
    dataset_dir, _ = write_processed_dataset(
        processed_root,
        repository,
        [make_normalized_issue(1), make_normalized_issue(2), make_normalized_issue(3)],
    )

    manifest = ProcessedManifest.model_validate_json(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )

    output_path = dataset_dir / "issues.jsonl"
    kept = output_path.read_text(encoding="utf-8").splitlines()[:2]
    output_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(DatasetReadError) as exc_info:
        list(read_dataset_issues(dataset_dir, manifest))
    assert "2 records" in str(exc_info.value)
    assert "3" in str(exc_info.value)
