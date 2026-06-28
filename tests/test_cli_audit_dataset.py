"""Tests for the audit-dataset CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.audit.builder import DEFAULT_AUDITS_ROOT, DEFAULT_PROCESSED_ROOT
from repotriage.cli import run_audit_dataset
from repotriage.github.models import RepositoryRef
from tests.helpers import make_normalized_issue, write_processed_dataset


class Args:
    def __init__(
        self, repo: str, dataset_id: str, processed_root: Path, audits_root: Path
    ) -> None:
        self.repo = repo
        self.dataset_id = dataset_id
        self.processed_root = processed_root
        self.audits_root = audits_root


def test_cli_success_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    _, dataset_id = write_processed_dataset(
        processed_root,
        repository,
        [make_normalized_issue(1, labels=["Bug"]), make_normalized_issue(2)],
    )

    exit_code = run_audit_dataset(
        Args("pandas-dev/pandas", dataset_id, processed_root, audits_root)
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Repository: pandas-dev/pandas" in captured.out
    assert f"Dataset ID: {dataset_id}" in captured.out
    assert f"Audit ID: {dataset_id}-a2" in captured.out
    assert "Issues analyzed: 2" in captured.out
    assert "Unique labels: 1" in captured.out
    assert "Labelled issues: 1" in captured.out
    assert "Warnings:" in captured.out
    assert "Audit-cache hit: no" in captured.out


def test_cli_cache_hit_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    _, dataset_id = write_processed_dataset(
        processed_root, repository, [make_normalized_issue(1)]
    )

    run_audit_dataset(Args("pandas-dev/pandas", dataset_id, processed_root, audits_root))
    capsys.readouterr()
    run_audit_dataset(Args("pandas-dev/pandas", dataset_id, processed_root, audits_root))
    captured = capsys.readouterr()
    assert "Audit-cache hit: yes" in captured.out


def test_cli_invalid_repo_returns_two(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_audit_dataset(
        Args(
            "not-a-repo",
            "20260101T000000000000Z-n1-aaaaaaaaaaaa",
            DEFAULT_PROCESSED_ROOT,
            DEFAULT_AUDITS_ROOT,
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "owner/name" in captured.err


def test_cli_invalid_dataset_id_returns_two(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_audit_dataset(
        Args(
            "pandas-dev/pandas",
            "not-a-valid-id",
            DEFAULT_PROCESSED_ROOT,
            DEFAULT_AUDITS_ROOT,
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Invalid dataset id" in captured.err


def test_cli_missing_dataset_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = run_audit_dataset(
        Args(
            "pandas-dev/pandas",
            "20260101T000000000000Z-n1-aaaaaaaaaaaa",
            tmp_path / "processed",
            tmp_path / "audits",
        )
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "No normalized dataset found" in captured.err
