"""Tests for atomic file writing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from repotriage.github.ingestion import atomic_write_bytes, atomic_write_json


def test_successful_write_produces_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "page.json"
    atomic_write_json(target, [{"id": 1, "title": "hello"}])
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == [{"id": 1, "title": "hello"}]


def test_unicode_issue_text_is_preserved(tmp_path: Path) -> None:
    target = tmp_path / "page.json"
    atomic_write_json(target, [{"title": "日本語テスト 🚀"}])
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload[0]["title"] == "日本語テスト 🚀"


def test_serialization_failure_does_not_create_final_file(tmp_path: Path) -> None:
    target = tmp_path / "page.json"

    with patch("repotriage.github.ingestion.json.dump", side_effect=TypeError("boom")):
        with pytest.raises(TypeError, match="boom"):
            atomic_write_json(target, object())

    assert not target.exists()
    leftovers = list(tmp_path.glob(".page.json.*"))
    assert leftovers == []


def test_replace_failure_preserves_existing_destination(tmp_path: Path) -> None:
    target = tmp_path / "page.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    with patch("repotriage.github.ingestion.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            atomic_write_bytes(target, b'{"new": true}\n')

    assert target.read_text(encoding="utf-8") == '{"old": true}\n'


def test_failure_cleans_up_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "page.json"

    with patch("repotriage.github.ingestion.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            atomic_write_bytes(target, b"[]\n")

    assert not target.exists()
    leftovers = list(tmp_path.glob(".page.json.*"))
    assert leftovers == []


def test_write_failure_raises_original_exception_when_cleanup_also_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "page.json"

    with (
        patch(
            "repotriage.github.ingestion._write_all_bytes",
            side_effect=OSError("write failed"),
        ),
        patch.object(Path, "unlink", side_effect=OSError("cleanup failed")),
    ):
        with pytest.raises(OSError, match="write failed") as exc_info:
            atomic_write_bytes(target, b"data")

    assert exc_info.value.args == ("write failed",)


def test_replace_failure_raises_original_exception_when_cleanup_also_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "page.json"

    with (
        patch(
            "repotriage.github.ingestion.os.replace",
            side_effect=OSError("replace failed"),
        ),
        patch.object(Path, "unlink", side_effect=OSError("cleanup failed")),
    ):
        with pytest.raises(OSError, match="replace failed") as exc_info:
            atomic_write_bytes(target, b"data")

    assert exc_info.value.args == ("replace failed",)
