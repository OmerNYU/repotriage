"""Tests for the RepoTriage CLI."""

from __future__ import annotations

import logging

import pytest

from repotriage.cli import run_fetch_issues
from repotriage.github.ingestion import DEFAULT_OUTPUT_ROOT
from repotriage.github.models import CacheConflictError


def test_expected_domain_error_is_printed_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)

    class Args:
        repo = "pandas-dev/pandas"
        max_pages = 3
        refresh = False
        output_root = DEFAULT_OUTPUT_ROOT

    def raise_conflict(*args: object, **kwargs: object) -> None:
        raise CacheConflictError(
            "Existing cache used max_pages=2, but the current request uses max_pages=3. "
            "Use --refresh to replace it."
        )

    monkeypatch.setattr("repotriage.cli.fetch_repository_issues", raise_conflict)

    exit_code = run_fetch_issues(Args())

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.count("max_pages=3") == 1
    assert "max_pages=3" not in caplog.text
