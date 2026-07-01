"""Backward-compatible re-exports of the normalized-issue JSONL reader.

The canonical implementation lives in :mod:`repotriage.dataset.reader`. This module
wraps the canonical reader so failures are raised as :class:`AuditReadError` for
historical ``except AuditReadError`` compatibility.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from repotriage.audit.models import AuditReadError
from repotriage.dataset.models import DatasetReadError, NormalizedIssue, ProcessedManifest
from repotriage.dataset.reader import iter_issues as _iter_issues
from repotriage.dataset.reader import read_dataset_issues as _read_dataset_issues

__all__ = ["AuditReadError", "iter_issues", "read_dataset_issues"]


def iter_issues(jsonl_path: Path) -> Iterator[NormalizedIssue]:
    """Yield validated normalized issues, raising :class:`AuditReadError` on failure."""
    try:
        yield from _iter_issues(jsonl_path)
    except DatasetReadError as exc:
        raise AuditReadError(str(exc)) from exc


def read_dataset_issues(
    dataset_dir: Path, manifest: ProcessedManifest
) -> Iterator[NormalizedIssue]:
    """Stream validated issues from a processed dataset, raising :class:`AuditReadError`."""
    try:
        yield from _read_dataset_issues(dataset_dir, manifest)
    except DatasetReadError as exc:
        raise AuditReadError(str(exc)) from exc
