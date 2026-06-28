"""Build (or reuse) an immutable audit artifact for one normalized dataset.

The audit binds itself to the normalized dataset bytes and lineage, computes objective
metrics and heuristic warnings, and publishes ``audit.json``, ``audit.md``, and
``manifest.json`` atomically from a hidden staging directory. Immutable audit ids are
never overwritten; an existing corrupt or incompatible artifact is reported, not
replaced.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from repotriage.audit.analyzer import Analyzer
from repotriage.audit.models import (
    AUDIT_DOCUMENT_SCHEMA_VERSION,
    AUDIT_JSON_FILE,
    AUDIT_MARKDOWN_FILE,
    AUDIT_VERSION,
    AuditCorruptionError,
    AuditDatasetError,
    AuditDocument,
    AuditError,
    AuditManifest,
    DatasetIdentity,
    compute_audit_id,
)
from repotriage.audit.policy import build_warnings
from repotriage.audit.reader import read_dataset_issues
from repotriage.audit.report import (
    serialize_audit_json,
    serialize_audit_markdown,
    sha256_hex,
)
from repotriage.dataset.builder import validate_processed_dataset_integrity
from repotriage.dataset.models import (
    ISSUE_SCHEMA_VERSION,
    ProcessedManifest,
    format_utc_datetime,
)
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.paths import resolve_within_directory

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_ROOT = Path("data/processed/github")
DEFAULT_AUDITS_ROOT = Path("data/audits/github")


@dataclass(frozen=True)
class AuditBuildResult:
    """Summary of an audit build or audit-cache hit."""

    repository: RepositoryRef
    audit_dir: Path
    manifest: AuditManifest
    document: AuditDocument
    cache_hit: bool


def _load_and_validate_dataset(
    repository: RepositoryRef, dataset_id: str, processed_root: Path
) -> tuple[Path, ProcessedManifest]:
    """Resolve and validate the normalized dataset's local integrity for auditing.

    Validation uses :func:`validate_processed_dataset_integrity`, which checks only the
    locally available processed artifact (manifest invariants, directory name, requested
    repository and dataset id, supported issue schema, safe paths, and the output
    SHA-256). It does not read raw GitHub pages or require the current raw cache, so an
    audit does not depend on the mutable raw source.
    """
    dataset_dir = processed_root / repository.slug / dataset_id
    if not dataset_dir.is_dir():
        raise AuditDatasetError(
            f"No normalized dataset found for {repository.full_name} with dataset id "
            f"{dataset_id!r} at {dataset_dir}."
        )

    manifest = validate_processed_dataset_integrity(
        dataset_dir,
        expected_repository=repository,
        expected_dataset_id=dataset_id,
        expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
    )
    return dataset_dir, manifest


def validate_audit_artifact(
    audit_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_dataset_id: str,
    expected_dataset_output_sha256: str,
    expected_issue_schema_version: str,
    expected_normalizer_version: str,
    expected_audit_id: str,
    check_dir_name: bool = True,
) -> tuple[AuditManifest, AuditDocument]:
    """Validate an on-disk audit artifact, raising on any corruption or mismatch.

    A lineage mismatch (versions, ids, or the bound dataset hash) or a report-hash
    mismatch (tampering) is reported as corruption and never treated as a cache hit.
    """
    manifest_path = audit_dir / "manifest.json"
    if not manifest_path.is_file():
        raise AuditCorruptionError(f"Missing audit manifest at {manifest_path}")
    try:
        manifest = AuditManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise AuditCorruptionError(f"Invalid audit manifest at {manifest_path}: {exc}") from exc
    except OSError as exc:
        raise AuditCorruptionError(
            f"Unable to read audit manifest at {manifest_path}: {exc}"
        ) from exc

    if check_dir_name and audit_dir.name != manifest.audit_id:
        raise AuditCorruptionError(
            f"Audit directory {audit_dir.name!r} does not match manifest audit_id "
            f"{manifest.audit_id!r}."
        )

    if manifest.audit_id != expected_audit_id:
        raise AuditCorruptionError(
            f"Audit manifest audit_id {manifest.audit_id!r} does not match expected "
            f"{expected_audit_id!r}."
        )
    if manifest.audit_version != AUDIT_VERSION:
        raise AuditCorruptionError(
            f"Audit manifest audit_version {manifest.audit_version!r} does not match the "
            f"current audit version {AUDIT_VERSION!r}."
        )
    if manifest.repository != expected_repository.full_name:
        raise AuditCorruptionError(
            f"Audit manifest repository {manifest.repository!r} does not match expected "
            f"{expected_repository.full_name!r}."
        )
    if manifest.dataset_id != expected_dataset_id:
        raise AuditCorruptionError(
            f"Audit manifest dataset_id {manifest.dataset_id!r} does not match expected "
            f"{expected_dataset_id!r}."
        )
    if manifest.dataset_output_sha256 != expected_dataset_output_sha256:
        raise AuditCorruptionError(
            "Audit manifest dataset_output_sha256 does not match the normalized dataset; "
            "the audited dataset bytes have changed."
        )
    if manifest.issue_schema_version != expected_issue_schema_version:
        raise AuditCorruptionError(
            f"Audit manifest issue_schema_version {manifest.issue_schema_version!r} does not "
            f"match expected {expected_issue_schema_version!r}."
        )
    if manifest.normalizer_version != expected_normalizer_version:
        raise AuditCorruptionError(
            f"Audit manifest normalizer_version {manifest.normalizer_version!r} does not match "
            f"expected {expected_normalizer_version!r}."
        )

    if manifest.audit_document_schema_version != AUDIT_DOCUMENT_SCHEMA_VERSION:
        raise AuditCorruptionError(
            f"Audit manifest audit_document_schema_version "
            f"{manifest.audit_document_schema_version!r} does not match the current "
            f"document schema {AUDIT_DOCUMENT_SCHEMA_VERSION!r}."
        )

    document = _verify_report_file(
        audit_dir,
        relative_path=manifest.audit_json_file,
        expected_sha256=manifest.audit_json_sha256,
        parse_document=True,
    )
    _verify_report_file(
        audit_dir,
        relative_path=manifest.audit_markdown_file,
        expected_sha256=manifest.audit_markdown_sha256,
        parse_document=False,
    )
    assert document is not None
    _cross_check_document_against_manifest(manifest, document)
    return manifest, document


def _cross_check_document_against_manifest(
    manifest: AuditManifest, document: AuditDocument
) -> None:
    """Verify ``audit.json`` agrees semantically with ``manifest.json``.

    Both files are individually hash-verified before this runs; this is accidental-
    corruption and local-consistency checking, not protection against a coordinated
    malicious rewrite of every file and hash together.
    """
    identity = document.identity
    checks: list[tuple[str, object, object]] = [
        ("audit_id", identity.audit_id, manifest.audit_id),
        ("repository", identity.repository, manifest.repository),
        ("dataset_id", identity.dataset_id, manifest.dataset_id),
        ("dataset_output_sha256", identity.dataset_output_sha256, manifest.dataset_output_sha256),
        ("audit_version", identity.audit_version, manifest.audit_version),
        (
            "audit_document_schema_version",
            document.schema_version,
            manifest.audit_document_schema_version,
        ),
        ("issue_schema_version", identity.issue_schema_version, manifest.issue_schema_version),
        ("normalizer_version", identity.normalizer_version, manifest.normalizer_version),
        (
            "issues_analyzed",
            document.repository_summary.total_issues,
            manifest.issues_analyzed,
        ),
    ]
    for field_name, document_value, manifest_value in checks:
        if document_value != manifest_value:
            raise AuditCorruptionError(
                f"audit.json {field_name} {document_value!r} disagrees with manifest "
                f"{field_name} {manifest_value!r}."
            )

    if document.schema_version != AUDIT_DOCUMENT_SCHEMA_VERSION:
        raise AuditCorruptionError(
            f"audit.json schema_version {document.schema_version!r} does not match the "
            f"current document schema {AUDIT_DOCUMENT_SCHEMA_VERSION!r}."
        )


def _verify_report_file(
    audit_dir: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    parse_document: bool,
) -> AuditDocument | None:
    try:
        resolved = resolve_within_directory(audit_dir, relative_path)
    except ValueError as exc:
        raise AuditCorruptionError(
            f"Audit manifest references an unsafe path: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise AuditCorruptionError(f"Missing audit report file: {relative_path}")
    data = resolved.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise AuditCorruptionError(
            f"Audit report hash mismatch for {relative_path}: expected {expected_sha256}, "
            f"found {actual}."
        )
    if not parse_document:
        return None
    try:
        return AuditDocument.model_validate_json(data.decode("utf-8"))
    except (ValidationError, UnicodeDecodeError) as exc:
        raise AuditCorruptionError(
            f"Invalid audit document in {relative_path}: {exc}"
        ) from exc


def publish_audit(staging_dir: Path, final_dir: Path) -> None:
    """Atomically publish a completed staging directory to its immutable final path."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise AuditError(f"Refusing to overwrite existing audit directory {final_dir}.")
    os.rename(staging_dir, final_dir)


def audit_dataset(
    repository: RepositoryRef,
    dataset_id: str,
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    audits_root: Path = DEFAULT_AUDITS_ROOT,
) -> AuditBuildResult:
    """Audit one explicit normalized dataset, publishing or reusing an immutable artifact."""
    dataset_dir, dataset_manifest = _load_and_validate_dataset(
        repository, dataset_id, processed_root
    )

    if dataset_manifest.issues_written == 0:
        raise AuditDatasetError(
            f"Cannot audit dataset {dataset_id}: it contains zero issues."
        )

    audit_id = compute_audit_id(dataset_id, AUDIT_VERSION)
    final_dir = audits_root / repository.slug / audit_id

    if final_dir.exists():
        manifest, document = validate_audit_artifact(
            final_dir,
            expected_repository=repository,
            expected_dataset_id=dataset_manifest.dataset_id,
            expected_dataset_output_sha256=dataset_manifest.output_sha256,
            expected_issue_schema_version=dataset_manifest.issue_schema_version,
            expected_normalizer_version=dataset_manifest.normalizer_version,
            expected_audit_id=audit_id,
        )
        logger.info("Audit-cache hit for %s at %s", repository.full_name, final_dir)
        return AuditBuildResult(
            repository=repository,
            audit_dir=final_dir,
            manifest=manifest,
            document=document,
            cache_hit=True,
        )

    analyzer = Analyzer()
    for issue in read_dataset_issues(dataset_dir, dataset_manifest):
        analyzer.add(issue)
    analysis = analyzer.finalize()

    identity = DatasetIdentity(
        audit_version=AUDIT_VERSION,
        audit_id=audit_id,
        repository=dataset_manifest.repository,
        dataset_id=dataset_manifest.dataset_id,
        dataset_output_sha256=dataset_manifest.output_sha256,
        issue_schema_version=dataset_manifest.issue_schema_version,
        normalizer_version=dataset_manifest.normalizer_version,
    )
    document = AuditDocument(
        identity=identity,
        repository_summary=analysis.repository_summary,
        text_metrics=analysis.text_metrics,
        label_metrics=analysis.label_metrics,
        temporal_metrics=analysis.temporal_metrics,
        warnings=build_warnings(analysis),
    )

    audit_json_bytes = serialize_audit_json(document)
    audit_markdown_bytes = serialize_audit_markdown(document)
    audit_json_sha256 = sha256_hex(audit_json_bytes)
    audit_markdown_sha256 = sha256_hex(audit_markdown_bytes)

    slug_dir = audits_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{repository.slug}.{audit_id}.staging-", dir=slug_dir)
    )

    logger.info(
        "Building audit %s for %s (%d issues analyzed)",
        audit_id,
        repository.full_name,
        analysis.repository_summary.total_issues,
    )

    try:
        atomic_write_bytes(staging_dir / AUDIT_JSON_FILE, audit_json_bytes)
        atomic_write_bytes(staging_dir / AUDIT_MARKDOWN_FILE, audit_markdown_bytes)

        written_json_sha = sha256_hex((staging_dir / AUDIT_JSON_FILE).read_bytes())
        written_md_sha = sha256_hex((staging_dir / AUDIT_MARKDOWN_FILE).read_bytes())
        if written_json_sha != audit_json_sha256 or written_md_sha != audit_markdown_sha256:
            raise AuditError("Audit report hash verification failed after writing staging files.")

        manifest = AuditManifest(
            audit_version=AUDIT_VERSION,
            audit_document_schema_version=AUDIT_DOCUMENT_SCHEMA_VERSION,
            audit_id=audit_id,
            repository=dataset_manifest.repository,
            dataset_id=dataset_manifest.dataset_id,
            dataset_output_sha256=dataset_manifest.output_sha256,
            issue_schema_version=dataset_manifest.issue_schema_version,
            normalizer_version=dataset_manifest.normalizer_version,
            built_at=datetime.now(UTC),
            issues_analyzed=analysis.repository_summary.total_issues,
            audit_json_file=AUDIT_JSON_FILE,
            audit_json_sha256=audit_json_sha256,
            audit_markdown_file=AUDIT_MARKDOWN_FILE,
            audit_markdown_sha256=audit_markdown_sha256,
        )
        atomic_write_bytes(
            staging_dir / "manifest.json",
            (manifest.model_dump_json() + "\n").encode("utf-8"),
        )

        validate_audit_artifact(
            staging_dir,
            expected_repository=repository,
            expected_dataset_id=dataset_manifest.dataset_id,
            expected_dataset_output_sha256=dataset_manifest.output_sha256,
            expected_issue_schema_version=dataset_manifest.issue_schema_version,
            expected_normalizer_version=dataset_manifest.normalizer_version,
            expected_audit_id=audit_id,
            check_dir_name=False,
        )

        publish_audit(staging_dir, final_dir)
    except BaseException:
        best_effort_remove_tree(staging_dir)
        raise

    logger.info("Published audit %s for %s at %s", audit_id, repository.full_name, final_dir)
    return AuditBuildResult(
        repository=repository,
        audit_dir=final_dir,
        manifest=manifest,
        document=document,
        cache_hit=False,
    )


def format_audit_summary(result: AuditBuildResult) -> str:
    """Build the user-facing audit summary."""
    manifest = result.manifest
    document = result.document
    summary = document.repository_summary

    if summary.earliest_created_at is not None and summary.latest_created_at is not None:
        date_range = (
            f"{format_utc_datetime(summary.earliest_created_at)} .. "
            f"{format_utc_datetime(summary.latest_created_at)}"
        )
    else:
        date_range = "n/a"

    lines = [
        f"Repository: {manifest.repository}",
        f"Dataset ID: {manifest.dataset_id}",
        f"Audit ID: {manifest.audit_id}",
        f"Issues analyzed: {manifest.issues_analyzed}",
        f"Unique labels: {document.label_metrics.unique_label_count}",
        f"Labelled issues: {summary.labelled_issues}",
        f"Date range: {date_range}",
        f"Warnings: {len(document.warnings)}",
        f"Output directory: {result.audit_dir}",
        f"Audit-cache hit: {'yes' if result.cache_hit else 'no'}",
    ]
    return "\n".join(lines)
