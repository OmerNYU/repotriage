"""Backward-compatible audit reader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.audit.models import AuditReadError
from repotriage.audit.reader import iter_issues, read_dataset_issues
from repotriage.dataset.builder import serialize_issues_jsonl
from repotriage.dataset.models import DatasetReadError
from tests.helpers import make_normalized_issue


def test_audit_reader_reexports() -> None:
    assert iter_issues is not None
    assert read_dataset_issues is not None
    assert issubclass(AuditReadError, DatasetReadError)


def test_malformed_json_raises_audit_read_error_via_audit_import_path(tmp_path: Path) -> None:
    good = serialize_issues_jsonl([make_normalized_issue(1)]).decode("utf-8").strip()
    path = tmp_path / "issues.jsonl"
    path.write_text(good + "\n{not valid json\n", encoding="utf-8")

    with pytest.raises(AuditReadError) as exc_info:
        list(iter_issues(path))
    assert "Malformed JSON" in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_blank_line_raises_audit_read_error_via_audit_import_path(tmp_path: Path) -> None:
    issues = [make_normalized_issue(1), make_normalized_issue(2)]
    serialized = serialize_issues_jsonl(issues).decode("utf-8").splitlines()
    path = tmp_path / "issues.jsonl"
    path.write_text(serialized[0] + "\n\n" + serialized[1] + "\n", encoding="utf-8")

    with pytest.raises(AuditReadError) as exc_info:
        list(iter_issues(path))
    assert "Blank line" in str(exc_info.value)
