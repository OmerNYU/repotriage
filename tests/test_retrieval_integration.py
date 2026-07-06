"""Real-data acceptance tests for the pandas retrieval-baseline artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.github.models import RepositoryRef
from repotriage.retrieval.builder import build_retrieval_baseline
from repotriage.retrieval.validators import validate_retrieval_against_model_dataset

_MODEL_DATASET_ID = "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7"
_CONFIG = Path("configs/retrieval_baselines/pandas-dev__pandas/tfidf-cosine-v1.json")
_MODEL_READY = Path("data/model_ready/github")


def _artifacts_present() -> bool:
    return (
        (_MODEL_READY / "pandas-dev__pandas" / _MODEL_DATASET_ID).is_dir()
        and _CONFIG.is_file()
    )


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_real_data_acceptance(tmp_path: Path) -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    retrieval_root = tmp_path / "retrieval_baselines"
    result = build_retrieval_baseline(
        repository,
        _MODEL_DATASET_ID,
        _CONFIG,
        model_ready_root=_MODEL_READY,
        retrieval_baselines_root=retrieval_root,
    )

    manifest = result.manifest
    assert manifest.model_dataset_id == _MODEL_DATASET_ID
    assert manifest.corpus_size == 840
    assert manifest.validation_query_count == 152
    assert manifest.test_query_count == 127
    assert manifest.target_count == 15
    assert "-rb1-" in manifest.retrieval_run_id
    assert manifest.index_semantic_sha256
    assert result.validation_metrics.all_zero_label_query_count == 25
    assert result.test_metrics.all_zero_label_query_count == 24

    validate_retrieval_against_model_dataset(
        result.retrieval_dir,
        _MODEL_READY / "pandas-dev__pandas" / _MODEL_DATASET_ID,
        expected_repository=repository,
        expected_model_dataset_id=_MODEL_DATASET_ID,
        config_path=_CONFIG,
    )

    second = build_retrieval_baseline(
        repository,
        _MODEL_DATASET_ID,
        _CONFIG,
        model_ready_root=_MODEL_READY,
        retrieval_baselines_root=retrieval_root,
    )
    assert second.cache_hit is True

    slug_dir = retrieval_root / repository.slug
    staging = [p for p in slug_dir.iterdir() if p.name.startswith(".") and "staging" in p.name]
    assert staging == []
