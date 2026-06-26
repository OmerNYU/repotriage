"""Tests for the dataset builder, publication, and processed-cache validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repotriage.dataset.builder import (
    build_dataset,
    compute_raw_snapshot_sha256,
    serialize_issues_jsonl,
)
from repotriage.dataset.models import (
    ISSUE_SCHEMA_VERSION,
    NORMALIZER_VERSION,
    DatasetBuildError,
    DatasetCorruptionError,
    DuplicateIssueError,
    MalformedIssueError,
    compute_dataset_id,
)
from repotriage.github.ingestion import validate_cache_integrity
from repotriage.github.models import RepositoryRef
from tests.helpers import make_raw_issue, make_raw_pull_request, write_raw_snapshot


def _expected_dataset_id(raw_root: Path, repository: RepositoryRef) -> str:
    cache_dir = raw_root / repository.slug
    manifest = validate_cache_integrity(cache_dir, expected_repository=repository)
    snapshot_hash = compute_raw_snapshot_sha256(cache_dir, manifest)
    return compute_dataset_id(manifest.fetched_at, NORMALIZER_VERSION, snapshot_hash)

FIXED_FETCHED_AT = datetime(2026, 6, 24, 16, 29, 50, 93080, tzinfo=UTC)


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture
def processed_root(tmp_path: Path) -> Path:
    return tmp_path / "processed"


def _read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_pull_requests_excluded_from_dataset(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1), make_raw_pull_request(2), make_raw_issue(3)]],
        fetched_at=FIXED_FETCHED_AT,
    )

    result = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    assert result.cache_hit is False
    assert result.manifest.raw_records_read == 3
    assert result.manifest.pull_requests_excluded == 1
    assert result.manifest.issues_written == 2

    rows = _read_jsonl(result.dataset_dir / "issues.jsonl")
    numbers = [row["issue_number"] for row in rows]
    assert numbers == [1, 3]


def test_count_reconciliation_holds(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1), make_raw_pull_request(2)], [make_raw_issue(3)]],
        fetched_at=FIXED_FETCHED_AT,
    )

    manifest = build_dataset(
        repository, raw_root=raw_root, processed_root=processed_root
    ).manifest

    assert (
        manifest.raw_records_read
        == manifest.pull_requests_excluded + manifest.issues_written
    )


def test_cross_check_against_raw_manifest_counts(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1), make_raw_issue(2)]], fetched_at=FIXED_FETCHED_AT
    )
    manifest_path = cache_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["issues_received"] = 1
    data["pull_requests_received"] = 1
    manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="pull_requests"):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)


def test_unlabelled_and_empty_body_counts(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [
            [
                make_raw_issue(1, labels=["Bug"], body="text"),
                make_raw_issue(2, labels=[], body=None),
                make_raw_issue(3, labels=[], body=""),
            ]
        ],
        fetched_at=FIXED_FETCHED_AT,
    )

    manifest = build_dataset(
        repository, raw_root=raw_root, processed_root=processed_root
    ).manifest
    assert manifest.unlabelled_issues == 2
    assert manifest.empty_body_issues == 2


def test_malformed_issue_fails_build_with_context(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    bad = make_raw_issue(7)
    del bad["title"]
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1), bad]], fetched_at=FIXED_FETCHED_AT
    )

    with pytest.raises(MalformedIssueError) as exc_info:
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    assert "page_0001.json" in str(exc_info.value)
    assert "issue 7" in str(exc_info.value)

    assert not (processed_root / repository.slug).exists() or list(
        (processed_root / repository.slug).iterdir()
    ) == []


def test_duplicate_issue_id_rejected(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1, issue_id=100), make_raw_issue(2, issue_id=100)]],
        fetched_at=FIXED_FETCHED_AT,
    )

    with pytest.raises(DuplicateIssueError, match="issue_id"):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)


def test_duplicate_issue_number_rejected(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(5, issue_id=1), make_raw_issue(5, issue_id=2)]],
        fetched_at=FIXED_FETCHED_AT,
    )

    with pytest.raises(DuplicateIssueError, match="issue_number"):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)


def test_issues_sorted_by_number_ascending(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(30)], [make_raw_issue(10), make_raw_issue(20)]],
        fetched_at=FIXED_FETCHED_AT,
    )

    result = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    rows = _read_jsonl(result.dataset_dir / "issues.jsonl")
    assert [row["issue_number"] for row in rows] == [10, 20, 30]


def test_deterministic_jsonl_bytes_and_hash(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(2, labels=["b", "a"]), make_raw_issue(1)]],
        fetched_at=FIXED_FETCHED_AT,
    )

    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    first_bytes = (first.dataset_dir / "issues.jsonl").read_bytes()
    first_hash = first.manifest.output_sha256

    import shutil

    shutil.rmtree(first.dataset_dir)
    second = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    second_bytes = (second.dataset_dir / "issues.jsonl").read_bytes()

    assert first.dataset_dir == second.dataset_dir
    assert first_bytes == second_bytes
    assert second.manifest.output_sha256 == first_hash


def test_jsonl_ends_with_newline_and_is_utf8(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1, title="日本語 🚀")]],
        fetched_at=FIXED_FETCHED_AT,
    )
    result = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    data = (result.dataset_dir / "issues.jsonl").read_bytes()
    assert data.endswith(b"\n")
    assert "日本語 🚀".encode() in data


def test_serialize_issues_jsonl_empty_is_empty_bytes() -> None:
    assert serialize_issues_jsonl([]) == b""


def test_manifest_lineage_fields(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    import hashlib

    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    expected_sha = hashlib.sha256(
        (cache_dir / "manifest.json").read_bytes()
    ).hexdigest()

    manifest = build_dataset(
        repository, raw_root=raw_root, processed_root=processed_root
    ).manifest

    expected_snapshot = compute_raw_snapshot_sha256(
        cache_dir,
        validate_cache_integrity(cache_dir, expected_repository=repository),
    )
    assert manifest.source_manifest_sha256 == expected_sha
    assert manifest.source_snapshot_sha256 == expected_snapshot
    assert manifest.source_manifest == f"{repository.slug}/manifest.json"
    assert manifest.source_manifest_schema_version == "2"
    assert manifest.issue_schema_version == ISSUE_SCHEMA_VERSION
    assert manifest.source_fetched_at == FIXED_FETCHED_AT
    assert manifest.source_api_version == "2026-03-10"
    assert manifest.source_pages_fetched == 1
    assert manifest.normalizer_version == NORMALIZER_VERSION
    assert manifest.dataset_id.endswith(f"-{expected_snapshot[:12]}")


def test_dataset_id_is_immutable_and_derived(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    expected_id = _expected_dataset_id(raw_root, repository)
    result = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    assert result.manifest.dataset_id == expected_id
    assert result.dataset_dir.name == expected_id


def test_new_raw_snapshot_produces_new_dataset_id(
    raw_root: Path, processed_root: Path
) -> None:
    repo_a = RepositoryRef(owner="o", name="a")
    repo_b = RepositoryRef(owner="o", name="b")
    write_raw_snapshot(
        raw_root,
        repo_a,
        [[make_raw_issue(1, repository=repo_a.full_name)]],
        fetched_at=FIXED_FETCHED_AT,
    )
    write_raw_snapshot(
        raw_root,
        repo_b,
        [[make_raw_issue(1, repository=repo_b.full_name)]],
        fetched_at=datetime(2026, 6, 25, 1, 2, 3, 4, tzinfo=UTC),
    )

    result_a = build_dataset(repo_a, raw_root=raw_root, processed_root=processed_root)
    result_b = build_dataset(repo_b, raw_root=raw_root, processed_root=processed_root)
    assert result_a.manifest.dataset_id != result_b.manifest.dataset_id


def test_repeated_build_returns_processed_cache_hit(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    assert first.cache_hit is False

    second = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    assert second.cache_hit is True
    assert second.dataset_dir == first.dataset_dir
    assert second.manifest.built_at == first.manifest.built_at


def test_missing_output_file_is_detected(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    (first.dataset_dir / "issues.jsonl").unlink()

    with pytest.raises(DatasetCorruptionError, match="Missing dataset output file"):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)


def test_output_hash_mismatch_is_detected(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    (first.dataset_dir / "issues.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(DatasetCorruptionError, match="hash mismatch"):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)


def test_unsafe_output_path_in_manifest_is_rejected(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    manifest_path = first.dataset_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["output_file"] = "../escape.jsonl"
    manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    with pytest.raises(DatasetCorruptionError):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)


def test_failure_leaves_no_published_dataset(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    bad = make_raw_issue(7)
    del bad["title"]
    write_raw_snapshot(raw_root, repository, [[bad]], fetched_at=FIXED_FETCHED_AT)

    with pytest.raises(MalformedIssueError):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    slug_dir = processed_root / repository.slug
    if slug_dir.exists():
        assert list(slug_dir.iterdir()) == []


def test_keyboard_interrupt_cleans_staging(
    repository: RepositoryRef,
    raw_root: Path,
    processed_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    dataset_id = _expected_dataset_id(raw_root, repository)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("repotriage.dataset.builder.atomic_write_bytes", interrupt)

    with pytest.raises(KeyboardInterrupt):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    final_dir = processed_root / repository.slug / dataset_id
    assert not final_dir.exists()
    leftovers = list((processed_root / repository.slug).glob(".*staging-*"))
    assert leftovers == []


def test_publish_does_not_overwrite_existing_dataset(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    from repotriage.dataset.builder import publish_dataset

    final_dir = processed_root / repository.slug / "existing"
    final_dir.mkdir(parents=True)
    (final_dir / "marker").write_text("keep", encoding="utf-8")

    staging = processed_root / repository.slug / ".staging"
    staging.mkdir()

    with pytest.raises(DatasetBuildError, match="Refusing to overwrite"):
        publish_dataset(staging, final_dir)
    assert (final_dir / "marker").read_text(encoding="utf-8") == "keep"


def test_source_manifest_path_is_portable_across_roots(
    repository: RepositoryRef, tmp_path: Path
) -> None:
    root_a = tmp_path / "abs_root_one" / "raw"
    root_b = tmp_path / "different" / "deeper" / "raw"
    processed_a = tmp_path / "p_a"
    processed_b = tmp_path / "p_b"
    write_raw_snapshot(root_a, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT)
    write_raw_snapshot(root_b, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT)

    man_a = build_dataset(repository, raw_root=root_a, processed_root=processed_a).manifest
    man_b = build_dataset(repository, raw_root=root_b, processed_root=processed_b).manifest

    assert man_a.source_manifest == man_b.source_manifest == f"{repository.slug}/manifest.json"
    assert man_a.dataset_id == man_b.dataset_id


@pytest.mark.parametrize("bad_item", ["a string", 7, [1, 2, 3], None])
def test_non_dict_raw_item_fails_with_source_context(
    repository: RepositoryRef, raw_root: Path, processed_root: Path, bad_item: object
) -> None:
    cache_dir = write_raw_snapshot(
        raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT
    )
    # Inject a structurally invalid element directly into the page bytes.
    page = cache_dir / "pages" / "page_0001.json"
    page.write_text(
        json.dumps([make_raw_issue(1), bad_item], indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(MalformedIssueError) as exc_info:
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    message = str(exc_info.value)
    assert "page_0001.json" in message
    assert "position 1" in message

    slug_dir = processed_root / repository.slug
    if slug_dir.exists():
        assert list(slug_dir.iterdir()) == []


def test_processed_cache_lineage_version_mismatch_is_corruption(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT)
    first = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    manifest_path = first.dataset_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["normalizer_version"] = "999"
    manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    before = manifest_path.read_text(encoding="utf-8")

    with pytest.raises(DatasetCorruptionError):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    assert manifest_path.read_text(encoding="utf-8") == before


def test_existing_incompatible_immutable_snapshot_is_not_overwritten(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(raw_root, repository, [[make_raw_issue(1)]], fetched_at=FIXED_FETCHED_AT)
    dataset_id = _expected_dataset_id(raw_root, repository)

    # Simulate a legacy/incompatible snapshot already occupying the immutable id.
    final_dir = processed_root / repository.slug / dataset_id
    final_dir.mkdir(parents=True)
    (final_dir / "manifest.json").write_text('{"schema_version": "0"}\n', encoding="utf-8")
    (final_dir / "issues.jsonl").write_text("legacy\n", encoding="utf-8")
    marker = (final_dir / "issues.jsonl").read_text(encoding="utf-8")

    with pytest.raises(DatasetCorruptionError):
        build_dataset(repository, raw_root=raw_root, processed_root=processed_root)

    assert (final_dir / "issues.jsonl").read_text(encoding="utf-8") == marker


def test_canonical_datetimes_in_published_jsonl(
    repository: RepositoryRef, raw_root: Path, processed_root: Path
) -> None:
    write_raw_snapshot(
        raw_root,
        repository,
        [[make_raw_issue(1, created_at="2026-06-24T21:09:03+05:00", closed_at=None)]],
        fetched_at=FIXED_FETCHED_AT,
    )
    result = build_dataset(repository, raw_root=raw_root, processed_root=processed_root)
    row = _read_jsonl(result.dataset_dir / "issues.jsonl")[0]
    assert row["created_at"] == "2026-06-24T16:09:03Z"
    assert row["closed_at"] is None
