"""Tests for temporal split configuration parsing and hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotriage.model_dataset.config import config_sha256, load_split_config
from repotriage.model_dataset.models import ModelDatasetConfigError
from tests.helpers import write_temporal_split_config


def test_valid_config_parses(tmp_path: Path) -> None:
    path = write_temporal_split_config(tmp_path / "split.json")
    config, digest = load_split_config(path)
    assert config.config_schema_version == "1"
    assert config.repository == "pandas-dev/pandas"
    assert config.validation_start.isoformat().startswith("2026-02-01")
    assert len(digest) == 64


def test_canonical_hash_invariant_to_whitespace(tmp_path: Path) -> None:
    path_a = write_temporal_split_config(tmp_path / "a.json")
    config_a, hash_a = load_split_config(path_a)
    payload = json.loads(path_a.read_text(encoding="utf-8"))
    path_b = tmp_path / "b.json"
    path_b.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    config_b, hash_b = load_split_config(path_b)
    assert config_sha256(config_a) == config_sha256(config_b)
    assert hash_a == hash_b


def test_semantic_change_changes_hash(tmp_path: Path) -> None:
    path_a = write_temporal_split_config(tmp_path / "a.json")
    path_b = write_temporal_split_config(
        tmp_path / "b.json", test_start="2026-05-01T00:00:00Z"
    )
    _, hash_a = load_split_config(path_a)
    _, hash_b = load_split_config(path_b)
    assert hash_a != hash_b


def test_test_start_must_follow_validation_start(tmp_path: Path) -> None:
    path = write_temporal_split_config(
        tmp_path / "bad.json",
        validation_start="2026-05-01T00:00:00Z",
        test_start="2026-04-01T00:00:00Z",
    )
    with pytest.raises(ModelDatasetConfigError):
        load_split_config(path)
