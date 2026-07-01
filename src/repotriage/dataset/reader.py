"""Streaming reader for normalized issue JSONL datasets.

The reader treats the normalized JSONL as the trusted input contract: it reads UTF-8,
tracks one-based line numbers, rejects blank lines, parses each line as JSON, validates
every record as :class:`NormalizedIssue`, and never silently skips malformed records.
All failures report the dataset path and line number.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from repotriage.dataset.models import DatasetReadError, NormalizedIssue, ProcessedManifest


def iter_issues(jsonl_path: Path) -> Iterator[NormalizedIssue]:
    """Yield validated normalized issues from a JSONL file, one per non-empty line.

    Blank or whitespace-only lines, JSON syntax errors, and schema-validation failures
    each raise :class:`DatasetReadError` carrying the dataset path and one-based line
    number. Records are never silently skipped.
    """
    try:
        handle = jsonl_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise DatasetReadError(f"Unable to read dataset output {jsonl_path}: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                raise DatasetReadError(
                    f"Blank line in dataset {jsonl_path} at line {line_number}; "
                    "normalized JSONL must not contain empty lines."
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetReadError(
                    f"Malformed JSON in dataset {jsonl_path} at line {line_number}: {exc}"
                ) from exc
            try:
                yield NormalizedIssue.model_validate(payload)
            except ValidationError as exc:
                raise DatasetReadError(
                    f"Invalid normalized issue in dataset {jsonl_path} at line "
                    f"{line_number}: {exc}"
                ) from exc


def read_dataset_issues(
    dataset_dir: Path, manifest: ProcessedManifest
) -> Iterator[NormalizedIssue]:
    """Stream validated issues from a processed dataset, enforcing the manifest count.

    The yielded count is compared against ``manifest.issues_written``; a mismatch raises
    :class:`DatasetReadError`. The check runs after the stream is exhausted, so callers
    that consume the iterator lazily receive the records first and the count guarantee
    on completion.
    """
    output_path = dataset_dir / manifest.output_file
    seen = 0
    for issue in iter_issues(output_path):
        seen += 1
        yield issue
    if seen != manifest.issues_written:
        raise DatasetReadError(
            f"Dataset {output_path} contains {seen} records but the manifest declares "
            f"{manifest.issues_written}."
        )
