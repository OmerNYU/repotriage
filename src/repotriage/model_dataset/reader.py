"""Streaming reader for model-ready JSONL records."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from repotriage.model_dataset.models import (
    MODEL_READY_RECORD_SCHEMA_VERSION,
    ModelDatasetManifest,
    ModelDatasetReadError,
    ModelReadyRecord,
    SplitName,
)


def iter_model_ready_records(jsonl_path: Path) -> Iterator[ModelReadyRecord]:
    """Yield validated model-ready records from a JSONL file, one per non-empty line."""
    try:
        handle = jsonl_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ModelDatasetReadError(
            f"Unable to read model-ready output {jsonl_path}: {exc}"
        ) from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                raise ModelDatasetReadError(
                    f"Blank line in model-ready dataset {jsonl_path} at line {line_number}; "
                    "JSONL must not contain empty lines."
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelDatasetReadError(
                    f"Malformed JSON in model-ready dataset {jsonl_path} at line "
                    f"{line_number}: {exc}"
                ) from exc
            try:
                yield ModelReadyRecord.model_validate(payload)
            except ValidationError as exc:
                raise ModelDatasetReadError(
                    f"Invalid model-ready record in {jsonl_path} at line "
                    f"{line_number}: {exc}"
                ) from exc


def read_model_ready_records(
    model_dataset_dir: Path,
    manifest: ModelDatasetManifest,
    *,
    split: SplitName | None = None,
) -> Iterator[ModelReadyRecord]:
    """Stream validated model-ready records, optionally filtered by split."""
    records_path = model_dataset_dir / manifest.records_file
    seen = 0
    for record in iter_model_ready_records(records_path):
        if record.schema_version != MODEL_READY_RECORD_SCHEMA_VERSION:
            raise ModelDatasetReadError(
                f"Unsupported record schema_version {record.schema_version!r} in {records_path}"
            )
        if split is not None and record.split != split:
            continue
        seen += 1
        yield record

    if split is None and seen != manifest.records_written:
        raise ModelDatasetReadError(
            f"Model-ready dataset {records_path} contains {seen} records but the manifest "
            f"declares {manifest.records_written}."
        )
