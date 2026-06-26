"""Tests for the build-dataset CLI command."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repotriage.cli import run_build_dataset
from repotriage.dataset.builder import (
    DEFAULT_PROCESSED_ROOT,
    compute_raw_snapshot_sha256,
)
from repotriage.dataset.models import (
    NORMALIZER_VERSION,
    MalformedIssueError,
    compute_dataset_id,
)
from repotriage.github.ingestion import DEFAULT_OUTPUT_ROOT, validate_cache_integrity
from repotriage.github.models import RepositoryRef
from tests.helpers import make_raw_issue, make_raw_pull_request, write_raw_snapshot

FIXED_FETCHED_AT = datetime(2026, 6, 24, 16, 29, 50, 93080, tzinfo=UTC)


def _expected_dataset_id(raw_root: Path, repository: RepositoryRef) -> str:
    cache_dir = raw_root / repository.slug
    manifest = validate_cache_integrity(cache_dir, expected_repository=repository)
    snapshot_hash = compute_raw_snapshot_sha256(cache_dir, manifest)
    return compute_dataset_id(manifest.fetched_at, NORMALIZER_VERSION, snapshot_hash)


class Args:
    def __init__(self, repo: str, raw_root: Path, processed_root: Path) -> None:
        self.repo = repo
        self.raw_root = raw_root
        self.processed_root = processed_root


def test_cli_success_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1), make_raw_pull_request(2), make_raw_issue(3, labels=[])]],
        fetched_at=FIXED_FETCHED_AT,
    )

    dataset_id = _expected_dataset_id(raw_root, repository)
    exit_code = run_build_dataset(Args("pandas-dev/pandas", raw_root, processed_root))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Repository: pandas-dev/pandas" in captured.out
    assert f"Dataset ID: {dataset_id}" in captured.out
    assert "Raw records read: 3" in captured.out
    assert "Pull requests excluded: 1" in captured.out
    assert "Issues written: 2" in captured.out
    assert "Unlabelled issues:" in captured.out
    assert "Empty-body issues:" in captured.out
    assert "Output directory:" in captured.out
    assert "Processed-cache hit: no" in captured.out


def test_cli_invalid_repo_returns_two(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_build_dataset(
        Args("not-a-repo", DEFAULT_OUTPUT_ROOT, DEFAULT_PROCESSED_ROOT)
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "owner/name" in captured.err


def test_cli_domain_error_printed_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)

    def raise_malformed(*args: object, **kwargs: object) -> None:
        raise MalformedIssueError("Malformed issue record at pages/page_0001.json position 0")

    monkeypatch.setattr("repotriage.cli.build_dataset", raise_malformed)

    exit_code = run_build_dataset(
        Args("pandas-dev/pandas", DEFAULT_OUTPUT_ROOT, DEFAULT_PROCESSED_ROOT)
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.count("Malformed issue record") == 1


def test_cli_cache_hit_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )

    run_build_dataset(Args("pandas-dev/pandas", raw_root, processed_root))
    capsys.readouterr()
    run_build_dataset(Args("pandas-dev/pandas", raw_root, processed_root))
    captured = capsys.readouterr()
    assert "Processed-cache hit: yes" in captured.out
