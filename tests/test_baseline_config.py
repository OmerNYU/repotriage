"""Tests for baseline configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.baseline.config import (
    config_semantic_sha256,
    config_source_sha256,
    load_baseline_config,
)
from repotriage.baseline.models import BaselineConfigError
from tests.helpers import write_baseline_config


def test_load_valid_config(tmp_path: Path) -> None:
    path = write_baseline_config(tmp_path / "baseline.json")
    config, raw, source_hash, semantic_hash = load_baseline_config(path)
    assert config.repository == "pandas-dev/pandas"
    assert len(config.candidates) == 3
    assert source_hash == config_source_sha256(raw)
    assert semantic_hash == config_semantic_sha256(config)


def test_semantic_hash_ignores_whitespace(tmp_path: Path) -> None:
    path_a = write_baseline_config(tmp_path / "a.json")
    path_b = write_baseline_config(tmp_path / "b.json")
    _, raw_a, source_a, semantic_a = load_baseline_config(path_a)
    pretty = path_b.read_text(encoding="utf-8").replace(
        '"random_state": 42', '\n  "random_state": 42'
    )
    path_b.write_text(pretty, encoding="utf-8")
    _, raw_b, source_b, semantic_b = load_baseline_config(path_b)
    assert semantic_a == semantic_b
    assert source_a != source_b


def test_duplicate_candidate_ids_rejected(tmp_path: Path) -> None:
    path = write_baseline_config(tmp_path / "baseline.json")
    payload = path.read_text(encoding="utf-8").replace("c2_bigram", "c1_unigram")
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(BaselineConfigError, match="unique"):
        load_baseline_config(path)


def test_missing_file_reports_error(tmp_path: Path) -> None:
    with pytest.raises(BaselineConfigError, match="Unable to read"):
        load_baseline_config(tmp_path / "missing.json")
