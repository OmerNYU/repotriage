"""Orchestrate GitHub issue fetching, caching, and manifest creation."""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from repotriage.github.client import GitHubClient
from repotriage.github.models import (
    DEFAULT_ISSUE_REQUEST_PARAMETERS,
    GITHUB_API_VERSION,
    CacheConflictError,
    CacheCorruptionError,
    CacheRecoveryError,
    IssueRequestParameters,
    Manifest,
    RepositoryRef,
    count_item_types,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data/raw/github")

PathRename = Callable[[Path, Path], None]


@dataclass(frozen=True)
class FetchResult:
    """Summary of a fetch or cache-hit operation."""

    repository: RepositoryRef
    cache_dir: Path
    manifest: Manifest
    cache_hit: bool


def cache_dir_for(repository: RepositoryRef, output_root: Path) -> Path:
    return output_root / repository.slug


def create_staging_directory(output_root: Path, repository: RepositoryRef) -> Path:
    """Create a unique staging directory for one fetch run."""
    output_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{repository.slug}.staging-", dir=output_root)
    )


def create_backup_path(cache_dir: Path) -> Path:
    """Create a unique backup path that does not already exist."""
    while True:
        candidate = cache_dir.with_name(f".{cache_dir.name}.backup-{secrets.token_hex(8)}")
        if not candidate.exists():
            return candidate


def _best_effort_remove(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        logger.warning("Failed to remove temporary file %s", path, exc_info=True)


def _best_effort_remove_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to remove staging directory %s", path, exc_info=True)


def _write_all_bytes(file_obj: Any, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = file_obj.write(data[offset:])
        if written is None or written == 0:
            raise OSError("Failed to write complete payload to temporary file")
        offset += written


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_file_path = Path(tmp_path)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            _write_all_bytes(tmp_file, data)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        _best_effort_remove(tmp_file_path)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically.

    Raw decoded API records with no fields removed or transformed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_file_path = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        _best_effort_remove(tmp_file_path)
        raise


def _read_manifest_file(cache_dir: Path) -> Manifest:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CacheCorruptionError(f"Missing manifest at {manifest_path}")
    try:
        return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise CacheCorruptionError(f"Invalid manifest at {manifest_path}: {exc}") from exc
    except OSError as exc:
        raise CacheCorruptionError(f"Unable to read manifest at {manifest_path}: {exc}") from exc


def _resolve_output_file(cache_dir: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise CacheCorruptionError(f"Output file path must be relative: {relative_path!r}")
    candidate = (cache_dir / relative_path).resolve()
    cache_resolved = cache_dir.resolve()
    try:
        candidate.relative_to(cache_resolved)
    except ValueError as exc:
        raise CacheCorruptionError(
            f"Output file path escapes cache directory: {relative_path!r}"
        ) from exc
    return candidate


def validate_cache(
    cache_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_max_pages: int,
    expected_api_version: str,
    expected_request_parameters: IssueRequestParameters,
) -> Manifest:
    """Validate that an on-disk cache matches the current request."""
    if not cache_dir.is_dir():
        raise CacheCorruptionError(f"Cache directory does not exist: {cache_dir}")

    manifest = _read_manifest_file(cache_dir)

    if manifest.repository != expected_repository.full_name:
        raise CacheConflictError(
            f"Existing cache is for {manifest.repository!r}, "
            f"but the current request uses {expected_repository.full_name!r}. "
            "Use --refresh to replace it."
        )

    if manifest.endpoint != expected_repository.issues_base_endpoint:
        raise CacheConflictError(
            f"Existing cache endpoint {manifest.endpoint!r} does not match the current "
            f"endpoint {expected_repository.issues_base_endpoint!r}. "
            "Use --refresh to replace it."
        )

    if manifest.api_version != expected_api_version:
        raise CacheConflictError(
            f"Existing cache uses api_version={manifest.api_version!r}, "
            f"but the current request uses api_version={expected_api_version!r}. "
            "Use --refresh to replace it."
        )

    if manifest.request_parameters != expected_request_parameters:
        raise CacheConflictError(
            "Existing cache request_parameters do not match the current request. "
            "Use --refresh to replace it."
        )

    if manifest.requested_max_pages != expected_max_pages:
        raise CacheConflictError(
            f"Existing cache used max_pages={manifest.requested_max_pages}, "
            f"but the current request uses max_pages={expected_max_pages}. "
            "Use --refresh to replace it."
        )

    if len(manifest.output_files) != manifest.pages_fetched:
        raise CacheCorruptionError(
            "Manifest output_files count does not match pages_fetched "
            f"({len(manifest.output_files)} != {manifest.pages_fetched})."
        )

    seen_paths: set[str] = set()
    for relative_path in manifest.output_files:
        if relative_path in seen_paths:
            raise CacheCorruptionError(f"Duplicate output file path in manifest: {relative_path!r}")
        seen_paths.add(relative_path)
        output_path = _resolve_output_file(cache_dir, relative_path)
        if not output_path.is_file():
            raise CacheCorruptionError(f"Missing cached output file: {relative_path}")

    return manifest


def _rollback_publication(
    *,
    cache_dir: Path,
    backup_dir: Path,
    rename: PathRename,
    publication_error: BaseException,
) -> None:
    try:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        rename(backup_dir, cache_dir)
    except BaseException:
        raise CacheRecoveryError(
            "Failed to publish new cache and automatic rollback also failed. "
            f"The previous valid cache remains at {backup_dir}. "
            f"Move it back to {cache_dir} manually."
        ) from publication_error
    raise publication_error


def publish_staging_directory(
    staging_dir: Path,
    cache_dir: Path,
    *,
    backup_dir: Path | None = None,
    rename: PathRename | None = None,
) -> None:
    """Publish a completed staging directory to the live cache path."""
    rename_path = rename or Path.rename
    created_backup = backup_dir
    backup_moved = False

    if not staging_dir.is_dir():
        raise CacheCorruptionError(f"Staging directory does not exist: {staging_dir}")

    if created_backup is None and cache_dir.exists():
        created_backup = create_backup_path(cache_dir)

    try:
        if cache_dir.exists():
            if created_backup is None:
                raise CacheCorruptionError("Backup directory is required to replace live cache")
            rename_path(cache_dir, created_backup)
            backup_moved = True
        rename_path(staging_dir, cache_dir)
    except BaseException as publication_error:
        if backup_moved and created_backup is not None and created_backup.exists():
            _rollback_publication(
                cache_dir=cache_dir,
                backup_dir=created_backup,
                rename=rename_path,
                publication_error=publication_error,
            )
        raise
    else:
        if created_backup is not None and created_backup.exists():
            shutil.rmtree(created_backup)


def fetch_repository_issues(
    repository: RepositoryRef,
    *,
    max_pages: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    refresh: bool = False,
    request_parameters: IssueRequestParameters = DEFAULT_ISSUE_REQUEST_PARAMETERS,
    client: GitHubClient | None = None,
    rename: PathRename | None = None,
) -> FetchResult:
    """Fetch repository issues into the local raw cache or return a cache hit."""
    cache_dir = cache_dir_for(repository, output_root)

    if not refresh and cache_dir.exists():
        manifest = validate_cache(
            cache_dir,
            expected_repository=repository,
            expected_max_pages=max_pages,
            expected_api_version=GITHUB_API_VERSION,
            expected_request_parameters=request_parameters,
        )
        logger.info("Cache hit for %s at %s", repository.full_name, cache_dir)
        return FetchResult(
            repository=repository,
            cache_dir=cache_dir,
            manifest=manifest,
            cache_hit=True,
        )

    staging_dir = create_staging_directory(output_root, repository)
    owns_client = client is None
    active_client = client or GitHubClient()

    logger.info(
        "Starting fetch for %s (max_pages=%d, refresh=%s)",
        repository.full_name,
        max_pages,
        refresh,
    )

    try:
        pages_dir = staging_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        output_files: list[str] = []
        raw_items_received = 0
        issues_received = 0
        pull_requests_received = 0
        pages_fetched = 0
        last_rate_limit = None

        for page in active_client.fetch_issues_pages(
            repository,
            max_pages=max_pages,
            request_parameters=request_parameters,
        ):
            page_path = pages_dir / f"page_{page.page_number:04d}.json"
            atomic_write_json(page_path, page.items)

            page_raw, page_issues, page_prs = count_item_types(page.items)
            raw_items_received += page_raw
            issues_received += page_issues
            pull_requests_received += page_prs
            pages_fetched += 1
            output_files.append(f"pages/{page_path.name}")
            last_rate_limit = page.rate_limit

            logger.info(
                "Page %d for %s: %d raw entries (%d issues, %d pull requests)",
                page.page_number,
                repository.full_name,
                page_raw,
                page_issues,
                page_prs,
            )

        manifest = Manifest(
            repository=repository.full_name,
            endpoint=repository.issues_base_endpoint,
            request_parameters=request_parameters,
            fetched_at=datetime.now(UTC),
            api_version=GITHUB_API_VERSION,
            authenticated=active_client.authenticated,
            requested_max_pages=max_pages,
            pages_fetched=pages_fetched,
            raw_items_received=raw_items_received,
            issues_received=issues_received,
            pull_requests_received=pull_requests_received,
            output_files=output_files,
            rate_limit_limit=last_rate_limit.limit if last_rate_limit else None,
            rate_limit_remaining=last_rate_limit.remaining if last_rate_limit else None,
            rate_limit_reset=last_rate_limit.reset if last_rate_limit else None,
        )
        atomic_write_json(staging_dir / "manifest.json", manifest.model_dump(mode="json"))
        publish_staging_directory(staging_dir, cache_dir, rename=rename)

        logger.info("Completed fetch for %s at %s", repository.full_name, cache_dir)
        return FetchResult(
            repository=repository,
            cache_dir=cache_dir,
            manifest=manifest,
            cache_hit=False,
        )
    except BaseException:
        _best_effort_remove_tree(staging_dir)
        raise
    finally:
        if owns_client:
            active_client.close()


def format_summary(result: FetchResult) -> str:
    """Build the user-facing completion summary."""
    manifest = result.manifest
    lines = [
        f"Repository: {manifest.repository}",
        f"Pages fetched: {manifest.pages_fetched}",
        f"Issues: {manifest.issues_received}",
        f"Pull requests: {manifest.pull_requests_received}",
        f"Cache: {result.cache_dir}",
    ]
    if result.cache_hit:
        lines.insert(0, "Cache hit: using existing import.")
    return "\n".join(lines)
