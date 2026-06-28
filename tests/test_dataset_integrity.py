"""Tests for processed-dataset integrity validation independent of the raw cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotriage.dataset.builder import validate_processed_dataset_integrity
from repotriage.dataset.models import ISSUE_SCHEMA_VERSION, DatasetCorruptionError
from repotriage.github.models import RepositoryRef
from tests.helpers import make_normalized_issue, write_processed_dataset


@pytest.fixture
def repository() -> RepositoryRef:
    return RepositoryRef(owner="pandas-dev", name="pandas")


def _write(tmp_path: Path, repository: RepositoryRef) -> tuple[Path, str]:
    processed_root = tmp_path / "processed"
    dataset_dir, dataset_id = write_processed_dataset(
        processed_root,
        repository,
        [make_normalized_issue(1, labels=["Bug"]), make_normalized_issue(2)],
    )
    return dataset_dir, dataset_id


def test_integrity_succeeds_without_raw_cache(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    dataset_dir, dataset_id = _write(tmp_path, repository)

    manifest = validate_processed_dataset_integrity(
        dataset_dir,
        expected_repository=repository,
        expected_dataset_id=dataset_id,
        expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
    )

    assert manifest.dataset_id == dataset_id
    assert manifest.repository == repository.full_name


def test_integrity_detects_output_tampering(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    dataset_dir, dataset_id = _write(tmp_path, repository)
    output_path = dataset_dir / "issues.jsonl"
    output_path.write_bytes(output_path.read_bytes() + b"\n")

    with pytest.raises(DatasetCorruptionError, match="hash mismatch"):
        validate_processed_dataset_integrity(
            dataset_dir,
            expected_repository=repository,
            expected_dataset_id=dataset_id,
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
        )


def test_integrity_rejects_wrong_repository(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    dataset_dir, dataset_id = _write(tmp_path, repository)
    other = RepositoryRef(owner="numpy", name="numpy")

    with pytest.raises(DatasetCorruptionError, match="repository"):
        validate_processed_dataset_integrity(
            dataset_dir,
            expected_repository=other,
            expected_dataset_id=dataset_id,
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
        )


def test_integrity_rejects_wrong_requested_dataset_id(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    dataset_dir, _ = _write(tmp_path, repository)

    with pytest.raises(DatasetCorruptionError, match="dataset_id"):
        validate_processed_dataset_integrity(
            dataset_dir,
            expected_repository=repository,
            expected_dataset_id="20990101T000000000000Z-n1-ffffffffffff",
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
            check_dir_name=False,
        )


def test_integrity_rejects_wrong_issue_schema_version(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    dataset_dir, dataset_id = _write(tmp_path, repository)

    with pytest.raises(DatasetCorruptionError, match="issue_schema_version"):
        validate_processed_dataset_integrity(
            dataset_dir,
            expected_repository=repository,
            expected_dataset_id=dataset_id,
            expected_issue_schema_version="999",
        )


def test_integrity_rejects_unsafe_output_path(
    tmp_path: Path, repository: RepositoryRef
) -> None:
    dataset_dir, dataset_id = _write(tmp_path, repository)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_file"] = "../escape.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(DatasetCorruptionError, match="unsafe"):
        validate_processed_dataset_integrity(
            dataset_dir,
            expected_repository=repository,
            expected_dataset_id=dataset_id,
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
        )
