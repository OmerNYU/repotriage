"""Tests for retrieval-baseline builder and cache behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.model_dataset.builder import build_model_dataset
from repotriage.retrieval.builder import build_retrieval_baseline, publish_retrieval_baseline
from repotriage.retrieval.models import RetrievalBuildError, RetrievalCorruptionError
from repotriage.retrieval.validators import validate_retrieval_artifact_integrity
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def _build(tmp_path: Path):
    fixture = _setup(tmp_path)
    model_result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    config_path = write_retrieval_baseline_config(tmp_path / "retrieval.json", min_df=1)
    retrieval_root = tmp_path / "retrieval_baselines"
    result = build_retrieval_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        retrieval_baselines_root=retrieval_root,
    )
    return fixture, model_result, config_path, retrieval_root, result


def test_builder_success_and_cache_hit(tmp_path: Path) -> None:
    fixture, model_result, config_path, retrieval_root, first = _build(tmp_path)
    assert first.cache_hit is False
    assert "-rb1-" in first.manifest.retrieval_run_id
    assert first.manifest.corpus_size == len(first.manifest.train_issue_ids)
    assert first.manifest.validation_query_count > 0
    assert first.manifest.test_query_count > 0

    second = build_retrieval_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        retrieval_baselines_root=retrieval_root,
    )
    assert second.cache_hit is True
    assert second.manifest.retrieval_run_id == first.manifest.retrieval_run_id


def test_publish_refuses_overwrite(tmp_path: Path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RetrievalBuildError):
        publish_retrieval_baseline(staging, final)


def test_corrupt_manifest_rejected(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, _retrieval_root, result = _build(tmp_path)
    neighbors_path = result.retrieval_dir / "neighbors_validation.jsonl"
    lines = neighbors_path.read_text(encoding="utf-8").splitlines()
    payload = __import__("json").loads(lines[0])
    payload["neighbors"][0]["similarity"] = 2.0
    lines[0] = __import__("json").dumps(payload)
    neighbors_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalCorruptionError):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_retrieval_run_id=result.manifest.retrieval_run_id,
        )
