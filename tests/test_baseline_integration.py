"""Real-data acceptance tests for the pandas baseline artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotriage.baseline.builder import train_baseline, validate_baseline_artifact_integrity
from repotriage.baseline.evaluator import metrics_from_predictions
from repotriage.baseline.models import PredictionRecord
from repotriage.github.models import RepositoryRef

_MODEL_DATASET_ID = "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7"
_CONFIG = Path("configs/baselines/pandas-dev__pandas/tfidf-logreg-v1.json")
_MODEL_READY = Path("data/model_ready/github")
_BASELINES = Path("data/baselines/github")
_OLD_RUN_IDS = (
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl1-861657b0b733",
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl2-e99e776d860d",
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl2-82163c83109b",
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl3-3574a6f815c3",
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl3-dddcfad62737",
)


def _artifacts_present() -> bool:
    return (
        (_MODEL_READY / "pandas-dev__pandas" / _MODEL_DATASET_ID).is_dir()
        and _CONFIG.is_file()
    )


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_real_data_acceptance(tmp_path: Path) -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    baselines_root = tmp_path / "baselines"
    result = train_baseline(
        repository,
        _MODEL_DATASET_ID,
        _CONFIG,
        model_ready_root=_MODEL_READY,
        baselines_root=baselines_root,
    )

    manifest = result.manifest
    assert manifest.model_dataset_id == _MODEL_DATASET_ID
    assert manifest.validation_record_count == 152
    assert manifest.test_record_count == 127
    assert manifest.target_count == 15
    assert manifest.baseline_version == "4"
    assert "-bl4-" in manifest.baseline_run_id
    assert manifest.model_semantic_sha256
    assert manifest.model_semantic_contract_version == "1"
    assert manifest.environment.numerical_thread_limit == 1
    assert manifest.environment.numerical_backends is not None
    assert result.selected_candidate_id in {
        "c1_unigram",
        "c2_bigram",
        "c3_bigram_balanced",
    }

    val_predictions: list[PredictionRecord] = []
    val_path = result.baseline_dir / "predictions_validation.jsonl"
    for line in val_path.read_text(encoding="utf-8").splitlines():
        val_predictions.append(PredictionRecord.model_validate_json(line))
    assert len(val_predictions) == 152 * 3
    assert all(record.candidate_id is not None for record in val_predictions)
    assert all(len(record.score_vector or []) == 15 for record in val_predictions)

    test_predictions: list[PredictionRecord] = []
    test_path = result.baseline_dir / "predictions_test.jsonl"
    for line in test_path.read_text(encoding="utf-8").splitlines():
        test_predictions.append(PredictionRecord.model_validate_json(line))
    assert len(test_predictions) == 127

    labels = json.loads(
        (_MODEL_READY / "pandas-dev__pandas" / _MODEL_DATASET_ID / "label_map.json").read_text(
            encoding="utf-8"
        )
    )["labels"]
    stored_test_metrics = json.loads(
        (result.baseline_dir / "metrics_test.json").read_text(encoding="utf-8")
    )
    recomputed = metrics_from_predictions(
        labels=labels,
        records=test_predictions,
        split="test",
    )
    assert recomputed.aggregate.subset_accuracy == pytest.approx(
        stored_test_metrics["aggregate"]["subset_accuracy"], abs=1e-12
    )

    second = train_baseline(
        repository,
        _MODEL_DATASET_ID,
        _CONFIG,
        model_ready_root=_MODEL_READY,
        baselines_root=baselines_root,
    )
    assert second.cache_hit is True

    slug_dir = baselines_root / repository.slug
    staging = [p for p in slug_dir.iterdir() if p.name.startswith(".") and "staging" in p.name]
    assert staging == []

    validate_baseline_artifact_integrity(
        result.baseline_dir,
        expected_repository=repository,
        expected_baseline_run_id=manifest.baseline_run_id,
    )

    for old_run_id in _OLD_RUN_IDS:
        old_dir = _BASELINES / "pandas-dev__pandas" / old_run_id
        if old_dir.is_dir():
            assert old_dir.exists()
            assert result.baseline_dir != old_dir
