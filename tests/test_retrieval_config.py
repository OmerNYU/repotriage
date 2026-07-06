"""Tests for retrieval-baseline configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotriage.retrieval.config import config_semantic_sha256, load_retrieval_config
from repotriage.retrieval.models import RetrievalConfigError
from tests.helpers import write_retrieval_baseline_config


def test_load_config_success(tmp_path: Path) -> None:
    path = write_retrieval_baseline_config(tmp_path / "retrieval.json")
    config, raw, source_hash, semantic_hash = load_retrieval_config(path)
    assert config.repository == "pandas-dev/pandas"
    assert config.top_k == 10
    assert config.tfidf.ngram_range == (1, 2)
    assert len(source_hash) == 64
    assert semantic_hash == config_semantic_sha256(config)
    assert source_hash != semantic_hash or raw


def test_semantic_hash_ignores_whitespace(tmp_path: Path) -> None:
    path = write_retrieval_baseline_config(tmp_path / "retrieval.json")
    config_a, _, _, semantic_a = load_retrieval_config(path)
    compact = json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":"))
    path.write_text(compact, encoding="utf-8")
    config_b, _, _, semantic_b = load_retrieval_config(path)
    assert semantic_a == semantic_b
    assert config_a.model_dump() == config_b.model_dump()


def test_invalid_schema_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"config_schema_version": "1"}\n', encoding="utf-8")
    with pytest.raises(RetrievalConfigError):
        load_retrieval_config(path)


def test_unsupported_version_rejected(tmp_path: Path) -> None:
    path = write_retrieval_baseline_config(tmp_path / "retrieval.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["retrieval_baseline_version"] = "2"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalConfigError):
        load_retrieval_config(path)
