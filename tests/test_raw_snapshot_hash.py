"""Tests for full raw-snapshot hashing and tamper detection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repotriage.dataset.builder import (
    build_dataset,
    compute_raw_snapshot_sha256,
)
from repotriage.dataset.models import DatasetBuildError, DatasetCorruptionError
from repotriage.github.ingestion import validate_cache_integrity
from repotriage.github.models import RepositoryRef
from tests.helpers import make_raw_issue, write_raw_snapshot

FIXED_FETCHED_AT = datetime(2026, 6, 24, 16, 29, 50, 93080, tzinfo=UTC)


@pytest.fixture
def repository() -> RepositoryRef:
    return RepositoryRef(owner="pandas-dev", name="pandas")


def _snapshot_hash(raw_root: Path, repository: RepositoryRef) -> str:
    cache_dir = raw_root / repository.slug
    manifest = validate_cache_integrity(cache_dir, expected_repository=repository)
    return compute_raw_snapshot_sha256(cache_dir, manifest)


def test_same_snapshot_produces_same_hash(tmp_path: Path, repository: RepositoryRef) -> None:
    raw_a = tmp_path / "a"
    raw_b = tmp_path / "b"
    pages = [[make_raw_issue(1), make_raw_issue(2)]]
    write_raw_snapshot(raw_a, repository, pages, fetched_at=FIXED_FETCHED_AT)
    write_raw_snapshot(raw_b, repository, pages, fetched_at=FIXED_FETCHED_AT)

    assert _snapshot_hash(raw_a, repository) == _snapshot_hash(raw_b, repository)


def test_changing_a_page_changes_hash(tmp_path: Path, repository: RepositoryRef) -> None:
    raw_root = tmp_path / "raw"
    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    before = _snapshot_hash(raw_root, repository)

    page = cache_dir / "pages" / "page_0001.json"
    data = json.loads(page.read_text(encoding="utf-8"))
    data[0]["title"] = "changed title"
    page.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    assert _snapshot_hash(raw_root, repository) != before


def test_reordering_output_files_changes_hash(tmp_path: Path, repository: RepositoryRef) -> None:
    raw_root = tmp_path / "raw"
    cache_dir = write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1)], [make_raw_issue(2)]],
        fetched_at=FIXED_FETCHED_AT,
    )
    manifest = validate_cache_integrity(cache_dir, expected_repository=repository)
    before = compute_raw_snapshot_sha256(cache_dir, manifest)

    reordered = manifest.model_copy(
        update={"output_files": list(reversed(manifest.output_files))}
    )
    assert compute_raw_snapshot_sha256(cache_dir, reordered) != before


def test_changing_unlisted_file_does_not_affect_hash(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    raw_root = tmp_path / "raw"
    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    before = _snapshot_hash(raw_root, repository)

    (cache_dir / "pages" / "page_9999.json").write_text("[]\n", encoding="utf-8")
    (cache_dir / "stray.txt").write_text("noise", encoding="utf-8")

    assert _snapshot_hash(raw_root, repository) == before


def test_unsafe_page_path_is_rejected_before_hashing(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    raw_root = tmp_path / "raw"
    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    manifest = validate_cache_integrity(cache_dir, expected_repository=repository)
    tampered = manifest.model_copy(update={"output_files": ["../escape.json"]})

    with pytest.raises(DatasetBuildError, match="Unsafe raw page path"):
        compute_raw_snapshot_sha256(cache_dir, tampered)


def test_repeated_build_detects_page_tamper_with_unchanged_manifest(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    original_output = (first.dataset_dir / "issues.jsonl").read_bytes()

    # Tamper a raw page WITHOUT changing manifest.json. To exercise the lineage
    # mismatch path, also rebuild under the original dataset directory by leaving
    # the manifest counts intact (title change keeps counts identical).
    page = cache_dir / "pages" / "page_0001.json"
    data = json.loads(page.read_text(encoding="utf-8"))
    data[0]["title"] = "tampered title"
    page.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    second = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    # Content-aware id: the tampered snapshot forks to a new immutable directory,
    # and the original snapshot is left untouched.
    assert second.dataset_dir != first.dataset_dir
    assert second.cache_hit is False
    assert (first.dataset_dir / "issues.jsonl").read_bytes() == original_output


def test_validate_detects_snapshot_mismatch_for_same_id(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    from repotriage.dataset.builder import validate_processed_dataset
    from repotriage.dataset.models import (
        ISSUE_SCHEMA_VERSION,
        NORMALIZER_VERSION,
        source_manifest_relpath,
    )

    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    write_raw_snapshot(raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT)
    result = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    with pytest.raises(DatasetCorruptionError, match="source_snapshot_sha256"):
        validate_processed_dataset(
            result.dataset_dir,
            expected_repository=repository,
            expected_normalizer_version=NORMALIZER_VERSION,
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
            expected_source_manifest_schema_version="2",
            expected_dataset_id=result.manifest.dataset_id,
            expected_source_manifest_relpath=source_manifest_relpath(repository.slug),
            expected_source_manifest_sha256=result.manifest.source_manifest_sha256,
            expected_source_snapshot_sha256="d" * 64,
        )
