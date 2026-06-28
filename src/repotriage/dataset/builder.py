"""Build an immutable normalized issue dataset from a validated raw snapshot."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from repotriage.dataset.models import (
    DEFAULT_OUTPUT_FILE,
    ISSUE_SCHEMA_VERSION,
    NORMALIZER_VERSION,
    DatasetBuildError,
    DatasetCorruptionError,
    DuplicateIssueError,
    MalformedIssueError,
    NormalizedIssue,
    ProcessedManifest,
    compute_dataset_id,
    source_manifest_relpath,
)
from repotriage.dataset.normalizer import normalize_issue
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.ingestion import (
    DEFAULT_OUTPUT_ROOT,
    validate_cache_integrity,
)
from repotriage.github.models import Manifest, RepositoryRef
from repotriage.paths import resolve_within_directory

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_ROOT = Path("data/processed/github")


@dataclass(frozen=True)
class DatasetBuildResult:
    """Summary of a dataset build or processed-cache hit."""

    repository: RepositoryRef
    dataset_dir: Path
    manifest: ProcessedManifest
    cache_hit: bool


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


RAW_SNAPSHOT_HASH_MARKER = b"repotriage-raw-snapshot-v1\0"
SOURCE_MANIFEST_LOGICAL_PATH = "manifest.json"


def _read_raw_file_bytes(raw_cache_dir: Path, relative_path: str) -> bytes:
    """Read a manifest-listed raw file's bytes after path-safety validation."""
    try:
        resolved = resolve_within_directory(raw_cache_dir, relative_path)
    except ValueError as exc:
        raise DatasetBuildError(f"Unsafe raw page path: {relative_path!r}") from exc
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise DatasetBuildError(f"Unable to read raw file {relative_path}: {exc}") from exc


def compute_raw_snapshot_sha256(raw_cache_dir: Path, raw_manifest: Manifest) -> str:
    """Hash the complete raw snapshot: manifest bytes plus every listed raw page.

    The digest is computed by feeding, in manifest order, a fixed version marker
    followed by each file's logical relative path and exact bytes, each prefixed by
    its byte length (8-byte big-endian). Length-prefixing makes path/content
    boundaries unambiguous so distinct snapshots cannot collide via concatenation.
    Only files listed in the ingestion manifest are hashed; every path is validated
    for safety before reading.
    """
    digest = hashlib.sha256()
    digest.update(RAW_SNAPSHOT_HASH_MARKER)

    def feed(logical_path: str, content: bytes) -> None:
        path_bytes = logical_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    manifest_bytes = (raw_cache_dir / "manifest.json").read_bytes()
    feed(SOURCE_MANIFEST_LOGICAL_PATH, manifest_bytes)
    for relative_path in raw_manifest.output_files:
        feed(relative_path, _read_raw_file_bytes(raw_cache_dir, relative_path))

    return digest.hexdigest()


def serialize_issues_jsonl(issues: list[NormalizedIssue]) -> bytes:
    """Serialize normalized issues to deterministic UTF-8 JSON Lines bytes."""
    lines: list[str] = []
    for issue in issues:
        payload = issue.model_dump(mode="json")
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        lines.append(line + "\n")
    return "".join(lines).encode("utf-8")


def _read_page_items(raw_cache_dir: Path, relative_path: str) -> list[Any]:
    try:
        page_path = resolve_within_directory(raw_cache_dir, relative_path)
    except ValueError as exc:
        raise DatasetBuildError(f"Unsafe raw page path: {relative_path!r}") from exc
    try:
        items = json.loads(page_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(f"Unable to read raw page {relative_path}: {exc}") from exc
    if not isinstance(items, list):
        raise DatasetBuildError(
            f"Raw page {relative_path} must contain a JSON array, got {type(items).__name__}."
        )
    return items


def _normalize_snapshot(
    raw_cache_dir: Path,
    manifest: Manifest,
    repository: RepositoryRef,
) -> tuple[list[NormalizedIssue], int, int]:
    """Read every raw page and normalize issues, excluding pull requests."""
    issues: list[NormalizedIssue] = []
    raw_records_read = 0
    pull_requests_excluded = 0

    for relative_path in manifest.output_files:
        items = _read_page_items(raw_cache_dir, relative_path)
        for position, item in enumerate(items):
            raw_records_read += 1
            if not isinstance(item, dict):
                raise MalformedIssueError(
                    f"Malformed record at {relative_path} position {position}: "
                    f"expected a JSON object, got {type(item).__name__}."
                )
            if "pull_request" in item:
                pull_requests_excluded += 1
                continue
            issues.append(
                normalize_issue(
                    item,
                    repository=repository.full_name,
                    source_page=relative_path,
                    position=position,
                )
            )

    return issues, raw_records_read, pull_requests_excluded


def _reject_duplicates(issues: list[NormalizedIssue]) -> None:
    seen_ids: set[int] = set()
    seen_numbers: set[int] = set()
    for issue in issues:
        if issue.issue_id in seen_ids:
            raise DuplicateIssueError(f"Duplicate issue_id detected: {issue.issue_id}")
        if issue.issue_number in seen_numbers:
            raise DuplicateIssueError(f"Duplicate issue_number detected: {issue.issue_number}")
        seen_ids.add(issue.issue_id)
        seen_numbers.add(issue.issue_number)


def validate_processed_dataset_integrity(
    dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_dataset_id: str,
    expected_issue_schema_version: str,
    check_dir_name: bool = True,
) -> ProcessedManifest:
    """Validate the integrity of a locally available processed dataset snapshot.

    This checks only the locally available processed artifact and never reads raw
    GitHub pages, requires the current raw cache, or revalidates raw-source lineage. It
    confirms: the dataset directory and ``manifest.json`` exist and parse (enforcing the
    processed-manifest invariants), the directory name matches the manifest dataset id,
    the manifest dataset id and repository match the explicitly requested values, the
    issue schema version is supported, the output path is safe and present, and the
    actual output bytes hash to ``manifest.output_sha256``.
    """
    if not dataset_dir.is_dir():
        raise DatasetCorruptionError(f"Processed dataset directory does not exist: {dataset_dir}")

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetCorruptionError(f"Missing processed manifest at {manifest_path}")
    try:
        manifest = ProcessedManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        raise DatasetCorruptionError(
            f"Invalid processed manifest at {manifest_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise DatasetCorruptionError(
            f"Unable to read processed manifest at {manifest_path}: {exc}"
        ) from exc

    if check_dir_name and dataset_dir.name != manifest.dataset_id:
        raise DatasetCorruptionError(
            f"Dataset directory {dataset_dir.name!r} does not match "
            f"manifest dataset_id {manifest.dataset_id!r}."
        )

    if manifest.dataset_id != expected_dataset_id:
        raise DatasetCorruptionError(
            f"Processed manifest dataset_id {manifest.dataset_id!r} does not match "
            f"expected {expected_dataset_id!r}."
        )

    if manifest.repository != expected_repository.full_name:
        raise DatasetCorruptionError(
            f"Processed manifest repository {manifest.repository!r} does not match "
            f"expected {expected_repository.full_name!r}."
        )

    if manifest.issue_schema_version != expected_issue_schema_version:
        raise DatasetCorruptionError(
            f"Processed manifest issue_schema_version {manifest.issue_schema_version!r} does "
            f"not match expected {expected_issue_schema_version!r}."
        )

    try:
        output_path = resolve_within_directory(dataset_dir, manifest.output_file)
    except ValueError as exc:
        raise DatasetCorruptionError(
            f"Processed manifest output_file is unsafe: {manifest.output_file!r}"
        ) from exc

    if not output_path.is_file():
        raise DatasetCorruptionError(f"Missing dataset output file: {manifest.output_file}")

    actual_sha256 = _sha256_hex(output_path.read_bytes())
    if actual_sha256 != manifest.output_sha256:
        raise DatasetCorruptionError(
            f"Dataset output hash mismatch for {manifest.output_file}: "
            f"expected {manifest.output_sha256}, found {actual_sha256}."
        )

    return manifest


def validate_processed_dataset(
    dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_normalizer_version: str,
    expected_issue_schema_version: str,
    expected_source_manifest_schema_version: str,
    expected_dataset_id: str,
    expected_source_manifest_relpath: str,
    expected_source_manifest_sha256: str,
    expected_source_snapshot_sha256: str,
    check_dir_name: bool = True,
) -> ProcessedManifest:
    """Validate a processed dataset for the dataset builder's current-raw-source needs.

    This first runs :func:`validate_processed_dataset_integrity` (local artifact
    integrity) and then additionally verifies compatibility with the currently available
    raw source: the normalizer version and the raw-source lineage fields. A lineage
    mismatch (versions, paths, or hashes) is reported as corruption and is never treated
    as a cache hit. Callers must not overwrite the existing directory.
    """
    manifest = validate_processed_dataset_integrity(
        dataset_dir,
        expected_repository=expected_repository,
        expected_dataset_id=expected_dataset_id,
        expected_issue_schema_version=expected_issue_schema_version,
        check_dir_name=check_dir_name,
    )

    if manifest.normalizer_version != expected_normalizer_version:
        raise DatasetCorruptionError(
            f"Processed manifest normalizer_version {manifest.normalizer_version!r} does not "
            f"match expected {expected_normalizer_version!r}."
        )

    if manifest.source_manifest_schema_version != expected_source_manifest_schema_version:
        raise DatasetCorruptionError(
            "Processed manifest source_manifest_schema_version "
            f"{manifest.source_manifest_schema_version!r} does not match the current raw "
            f"manifest schema {expected_source_manifest_schema_version!r}."
        )

    if manifest.source_manifest != expected_source_manifest_relpath:
        raise DatasetCorruptionError(
            f"Processed manifest source_manifest {manifest.source_manifest!r} does not match "
            f"expected logical path {expected_source_manifest_relpath!r}."
        )

    if manifest.source_manifest_sha256 != expected_source_manifest_sha256:
        raise DatasetCorruptionError(
            "Processed manifest source_manifest_sha256 does not match the current raw manifest."
        )

    if manifest.source_snapshot_sha256 != expected_source_snapshot_sha256:
        raise DatasetCorruptionError(
            "Processed manifest source_snapshot_sha256 does not match the current raw snapshot; "
            "the raw pages may have changed since this dataset was built."
        )

    return manifest


def publish_dataset(staging_dir: Path, final_dir: Path) -> None:
    """Atomically publish a completed staging directory to its immutable final path.

    Immutable dataset ids are never overwritten. If the final directory already
    exists (for example, a concurrent build won the race), the build is rejected.
    """
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise DatasetBuildError(
            f"Refusing to overwrite existing dataset directory {final_dir}."
        )
    os.rename(staging_dir, final_dir)


def build_dataset(
    repository: RepositoryRef,
    *,
    raw_root: Path = DEFAULT_OUTPUT_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> DatasetBuildResult:
    """Build (or reuse) an immutable normalized dataset from a raw snapshot."""
    raw_cache_dir = raw_root / repository.slug
    raw_manifest = validate_cache_integrity(raw_cache_dir, expected_repository=repository)

    source_manifest_path = raw_cache_dir / "manifest.json"
    source_manifest_sha256 = _sha256_hex(source_manifest_path.read_bytes())
    source_snapshot_sha256 = compute_raw_snapshot_sha256(raw_cache_dir, raw_manifest)
    source_manifest_rel = source_manifest_relpath(repository.slug)
    dataset_id = compute_dataset_id(
        raw_manifest.fetched_at, NORMALIZER_VERSION, source_snapshot_sha256
    )
    final_dir = processed_root / repository.slug / dataset_id

    if final_dir.exists():
        manifest = validate_processed_dataset(
            final_dir,
            expected_repository=repository,
            expected_normalizer_version=NORMALIZER_VERSION,
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
            expected_source_manifest_schema_version=raw_manifest.schema_version,
            expected_dataset_id=dataset_id,
            expected_source_manifest_relpath=source_manifest_rel,
            expected_source_manifest_sha256=source_manifest_sha256,
            expected_source_snapshot_sha256=source_snapshot_sha256,
        )
        logger.info("Processed-cache hit for %s at %s", repository.full_name, final_dir)
        return DatasetBuildResult(
            repository=repository,
            dataset_dir=final_dir,
            manifest=manifest,
            cache_hit=True,
        )

    issues, raw_records_read, pull_requests_excluded = _normalize_snapshot(
        raw_cache_dir, raw_manifest, repository
    )
    _reject_duplicates(issues)
    issues.sort(key=lambda issue: issue.issue_number)

    issues_written = len(issues)
    unlabelled_issues = sum(1 for issue in issues if not issue.labels)
    empty_body_issues = sum(1 for issue in issues if issue.body == "")

    if raw_records_read != pull_requests_excluded + issues_written:
        raise DatasetBuildError(
            "Count reconciliation failed: raw_records_read "
            f"({raw_records_read}) != pull_requests_excluded ({pull_requests_excluded}) "
            f"+ issues_written ({issues_written})."
        )

    _cross_check_raw_manifest(
        raw_manifest,
        raw_records_read=raw_records_read,
        pull_requests_excluded=pull_requests_excluded,
        issues_written=issues_written,
    )

    output_bytes = serialize_issues_jsonl(issues)
    output_sha256 = _sha256_hex(output_bytes)

    processed_root_slug = processed_root / repository.slug
    processed_root_slug.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{repository.slug}.{dataset_id}.staging-",
            dir=processed_root_slug,
        )
    )

    logger.info(
        "Building dataset %s for %s (%d issues, %d pull requests excluded)",
        dataset_id,
        repository.full_name,
        issues_written,
        pull_requests_excluded,
    )

    try:
        output_path = staging_dir / DEFAULT_OUTPUT_FILE
        atomic_write_bytes(output_path, output_bytes)

        written_sha256 = _sha256_hex(output_path.read_bytes())
        if written_sha256 != output_sha256:
            raise DatasetBuildError(
                f"Output hash verification failed for {output_path}: "
                f"expected {output_sha256}, found {written_sha256}."
            )

        manifest = ProcessedManifest(
            dataset_id=dataset_id,
            repository=repository.full_name,
            normalizer_version=NORMALIZER_VERSION,
            built_at=datetime.now(UTC),
            source_manifest=source_manifest_rel,
            source_manifest_sha256=source_manifest_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            source_manifest_schema_version=raw_manifest.schema_version,
            source_fetched_at=raw_manifest.fetched_at,
            source_api_version=raw_manifest.api_version,
            source_pages_fetched=raw_manifest.pages_fetched,
            raw_records_read=raw_records_read,
            pull_requests_excluded=pull_requests_excluded,
            issues_written=issues_written,
            unlabelled_issues=unlabelled_issues,
            empty_body_issues=empty_body_issues,
            output_file=DEFAULT_OUTPUT_FILE,
            output_sha256=output_sha256,
        )
        atomic_write_bytes(
            staging_dir / "manifest.json",
            (manifest.model_dump_json() + "\n").encode("utf-8"),
        )

        validate_processed_dataset(
            staging_dir,
            expected_repository=repository,
            expected_normalizer_version=NORMALIZER_VERSION,
            expected_issue_schema_version=ISSUE_SCHEMA_VERSION,
            expected_source_manifest_schema_version=raw_manifest.schema_version,
            expected_dataset_id=dataset_id,
            expected_source_manifest_relpath=source_manifest_rel,
            expected_source_manifest_sha256=source_manifest_sha256,
            expected_source_snapshot_sha256=source_snapshot_sha256,
            check_dir_name=False,
        )

        publish_dataset(staging_dir, final_dir)
    except BaseException:
        best_effort_remove_tree(staging_dir)
        raise

    logger.info("Published dataset %s for %s at %s", dataset_id, repository.full_name, final_dir)
    return DatasetBuildResult(
        repository=repository,
        dataset_dir=final_dir,
        manifest=manifest,
        cache_hit=False,
    )


def _cross_check_raw_manifest(
    raw_manifest: Manifest,
    *,
    raw_records_read: int,
    pull_requests_excluded: int,
    issues_written: int,
) -> None:
    """Verify normalized counts agree with the raw manifest's recorded totals."""
    if raw_records_read != raw_manifest.raw_items_received:
        raise DatasetBuildError(
            f"raw_records_read ({raw_records_read}) does not match raw manifest "
            f"raw_items_received ({raw_manifest.raw_items_received})."
        )
    if pull_requests_excluded != raw_manifest.pull_requests_received:
        raise DatasetBuildError(
            f"pull_requests_excluded ({pull_requests_excluded}) does not match raw manifest "
            f"pull_requests_received ({raw_manifest.pull_requests_received})."
        )
    if issues_written != raw_manifest.issues_received:
        raise DatasetBuildError(
            f"issues_written ({issues_written}) does not match raw manifest "
            f"issues_received ({raw_manifest.issues_received})."
        )


def format_dataset_summary(result: DatasetBuildResult) -> str:
    """Build the user-facing build summary."""
    manifest = result.manifest
    lines = [
        f"Repository: {manifest.repository}",
        f"Dataset ID: {manifest.dataset_id}",
        f"Raw records read: {manifest.raw_records_read}",
        f"Pull requests excluded: {manifest.pull_requests_excluded}",
        f"Issues written: {manifest.issues_written}",
        f"Unlabelled issues: {manifest.unlabelled_issues}",
        f"Empty-body issues: {manifest.empty_body_issues}",
        f"Output directory: {result.dataset_dir}",
        f"Processed-cache hit: {'yes' if result.cache_hit else 'no'}",
    ]
    return "\n".join(lines)
