"""Tests for audit orchestration, publication, and immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.audit.builder import audit_dataset, publish_audit
from repotriage.audit.models import (
    AuditCorruptionError,
    AuditDatasetError,
    AuditError,
    compute_audit_id,
)
from repotriage.dataset.models import DatasetCorruptionError
from repotriage.github.models import RepositoryRef
from tests.helpers import make_normalized_issue, write_processed_dataset


@pytest.fixture
def repository() -> RepositoryRef:
    return RepositoryRef(owner="pandas-dev", name="pandas")


@pytest.fixture
def processed_root(tmp_path: Path) -> Path:
    return tmp_path / "processed"


@pytest.fixture
def audits_root(tmp_path: Path) -> Path:
    return tmp_path / "audits"


def _issues() -> list:
    return [
        make_normalized_issue(1, labels=["Bug"]),
        make_normalized_issue(2, labels=["Bug", "Docs"]),
        make_normalized_issue(3, labels=[]),
    ]


def test_explicit_selection_success(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())

    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )

    assert result.cache_hit is False
    assert result.audit_dir == audits_root / repository.slug / f"{dataset_id}-a2"
    assert (result.audit_dir / "audit.json").is_file()
    assert (result.audit_dir / "audit.md").is_file()
    assert (result.audit_dir / "manifest.json").is_file()
    assert result.manifest.issues_analyzed == 3


def test_audit_manifest_lineage(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    dataset_dir, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    from repotriage.dataset.models import ProcessedManifest

    dataset_manifest = ProcessedManifest.model_validate_json(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )

    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    manifest = result.manifest

    assert manifest.audit_id == compute_audit_id(dataset_id, "2")
    assert manifest.dataset_id == dataset_id
    assert manifest.dataset_output_sha256 == dataset_manifest.output_sha256
    assert manifest.issue_schema_version == dataset_manifest.issue_schema_version
    assert manifest.normalizer_version == dataset_manifest.normalizer_version
    assert manifest.issues_analyzed == 3


def test_repeated_audit_returns_cache_hit(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())

    first = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    second = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.audit_dir == first.audit_dir
    assert second.manifest.built_at == first.manifest.built_at


def test_tampered_audit_json_is_detected(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    (result.audit_dir / "audit.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(AuditCorruptionError, match="hash mismatch"):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )


def test_tampered_audit_markdown_is_detected(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    (result.audit_dir / "audit.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(AuditCorruptionError, match="hash mismatch"):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )


def test_existing_incompatible_audit_not_overwritten(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    audit_id = compute_audit_id(dataset_id, "2")
    final_dir = audits_root / repository.slug / audit_id
    final_dir.mkdir(parents=True)
    (final_dir / "manifest.json").write_text('{"schema_version": "0"}\n', encoding="utf-8")
    marker = (final_dir / "manifest.json").read_text(encoding="utf-8")

    with pytest.raises(AuditCorruptionError):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )

    assert (final_dir / "manifest.json").read_text(encoding="utf-8") == marker


def test_failure_leaves_no_final_directory_and_cleans_staging(
    repository: RepositoryRef,
    processed_root: Path,
    audits_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("repotriage.audit.builder.atomic_write_bytes", boom)

    with pytest.raises(RuntimeError, match="boom"):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )

    final_dir = audits_root / repository.slug / f"{dataset_id}-a2"
    assert not final_dir.exists()
    slug_dir = audits_root / repository.slug
    assert list(slug_dir.glob(".*staging-*")) == []


def test_keyboard_interrupt_cleans_staging(
    repository: RepositoryRef,
    processed_root: Path,
    audits_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("repotriage.audit.builder.atomic_write_bytes", interrupt)

    with pytest.raises(KeyboardInterrupt):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )

    final_dir = audits_root / repository.slug / f"{dataset_id}-a2"
    assert not final_dir.exists()
    slug_dir = audits_root / repository.slug
    assert list(slug_dir.glob(".*staging-*")) == []


def test_missing_dataset_raises_audit_dataset_error(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    dataset_id = "20260101T000000000000Z-n1-aaaaaaaaaaaa"
    with pytest.raises(AuditDatasetError):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )


def test_modified_dataset_bytes_detected_as_corruption(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    dataset_dir, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    output_path = dataset_dir / "issues.jsonl"
    output_path.write_bytes(output_path.read_bytes() + b"\n")

    with pytest.raises(DatasetCorruptionError):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )


def test_publish_audit_refuses_to_overwrite(tmp_path: Path) -> None:
    final_dir = tmp_path / "audit"
    final_dir.mkdir()
    (final_dir / "marker").write_text("keep", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(AuditError, match="Refusing to overwrite"):
        publish_audit(staging, final_dir)
    assert (final_dir / "marker").read_text(encoding="utf-8") == "keep"


def test_preexisting_a1_artifact_is_left_intact_when_building_a2(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    a1_dir = audits_root / repository.slug / f"{dataset_id}-a1"
    a1_dir.mkdir(parents=True)
    a1_manifest = '{"schema_version": "1", "audit_version": "1"}\n'
    (a1_dir / "manifest.json").write_text(a1_manifest, encoding="utf-8")

    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )

    assert result.audit_dir == audits_root / repository.slug / f"{dataset_id}-a2"
    assert result.audit_dir.is_dir()
    assert a1_dir.is_dir()
    assert (a1_dir / "manifest.json").read_text(encoding="utf-8") == a1_manifest


def test_empty_dataset_is_rejected_without_publishing(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, [])

    with pytest.raises(AuditDatasetError, match="zero issues"):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )

    final_dir = audits_root / repository.slug / f"{dataset_id}-a2"
    assert not final_dir.exists()
    slug_dir = audits_root / repository.slug
    if slug_dir.exists():
        assert list(slug_dir.glob(".*staging-*")) == []


def test_audit_succeeds_without_raw_cache_present(
    repository: RepositoryRef, processed_root: Path, audits_root: Path
) -> None:
    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())

    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )

    assert result.cache_hit is False
    assert result.manifest.issues_analyzed == 3


def _rehash_manifest_after_audit_json_edit(audit_dir: Path) -> None:
    """Recompute and rewrite the manifest hash for a locally edited ``audit.json``.

    This makes the per-file hash check pass so the semantic cross-check is exercised.
    """
    import hashlib
    import json

    audit_json = audit_dir / "audit.json"
    new_hash = hashlib.sha256(audit_json.read_bytes()).hexdigest()
    manifest_path = audit_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit_json_sha256"] = new_hash
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("field_path", "new_value"),
    [
        (("identity", "dataset_id"), "20990101T000000000000Z-n1-ffffffffffff"),
        (("identity", "repository"), "other/repo"),
        (("repository_summary", "total_issues"), 999),
    ],
)
def test_semantic_cross_check_detects_document_manifest_disagreement(
    repository: RepositoryRef,
    processed_root: Path,
    audits_root: Path,
    field_path: tuple[str, str],
    new_value: object,
) -> None:
    import json

    _, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    result = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )

    audit_json_path = result.audit_dir / "audit.json"
    document = json.loads(audit_json_path.read_text(encoding="utf-8"))
    section, key = field_path
    document[section][key] = new_value
    audit_json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rehash_manifest_after_audit_json_edit(result.audit_dir)

    with pytest.raises(AuditCorruptionError):
        audit_dataset(
            repository, dataset_id, processed_root=processed_root, audits_root=audits_root
        )
