"""Tests for the streaming model-ready JSONL reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.model_dataset.models import ModelDatasetReadError
from repotriage.model_dataset.reader import iter_model_ready_records, read_model_ready_records
from tests.test_model_dataset_builder import _setup


def test_valid_jsonl_streaming(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    from repotriage.model_dataset.builder import build_model_dataset

    result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    records_path = result.model_dataset_dir / "records.jsonl"
    records = list(iter_model_ready_records(records_path))
    assert len(records) == 10


def test_blank_line_rejected(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    from repotriage.model_dataset.builder import build_model_dataset

    result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    records_path = result.model_dataset_dir / "records.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines()
    records_path.write_text(lines[0] + "\n\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(ModelDatasetReadError, match="Blank line"):
        list(iter_model_ready_records(records_path))


def test_record_count_mismatch_against_manifest(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    from repotriage.model_dataset.builder import build_model_dataset

    result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    records_path = result.model_dataset_dir / "records.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines()[:2]
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ModelDatasetReadError, match="contains 2 records"):
        list(read_model_ready_records(result.model_dataset_dir, result.manifest))
