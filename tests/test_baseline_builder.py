"""Tests for baseline builder orchestration."""

from __future__ import annotations

import json

import pytest

from repotriage.baseline.builder import (
    train_baseline,
    validate_baseline_artifact_integrity,
    verify_baseline_model_consistency,
)
from repotriage.baseline.models import BaselineCorruptionError
from tests.helpers import write_baseline_config
from tests.test_model_dataset_builder import _setup


def _build_model_ready(tmp_path):
    fixture = _setup(tmp_path)
    from repotriage.model_dataset.builder import build_model_dataset

    model_result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    return fixture, model_result


def test_builder_success_and_cache_hit(tmp_path) -> None:
    fixture, model_result = _build_model_ready(tmp_path)
    config_path = write_baseline_config(tmp_path / "baseline.json")
    baselines_root = tmp_path / "baselines"
    first = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=baselines_root,
    )
    assert first.cache_hit is False
    assert first.manifest.baseline_version == "4"
    assert "-bl4-" in first.manifest.baseline_run_id
    assert first.manifest.model_semantic_sha256
    assert first.manifest.model_semantic_contract_version == "1"
    assert first.manifest.baseline_experiment_sha256
    assert first.manifest.numerical_environment_sha256
    assert first.manifest.environment.numerical_thread_limit == 1
    assert first.manifest.environment.numerical_backends is not None
    second = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=baselines_root,
    )
    assert second.cache_hit is True
    assert first.manifest.baseline_run_id == second.manifest.baseline_run_id


def test_identity_changes_with_seed(tmp_path) -> None:
    fixture, model_result = _build_model_ready(tmp_path)
    config_path = write_baseline_config(tmp_path / "baseline.json")
    config_text = config_path.read_text(encoding="utf-8")
    config_a = json.loads(config_text)
    config_b = json.loads(config_text)
    config_b["random_state"] = 99
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(config_a, indent=2) + "\n", encoding="utf-8")
    path_b.write_text(json.dumps(config_b, indent=2) + "\n", encoding="utf-8")
    result_a = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        path_a,
        model_ready_root=fixture.model_ready_root,
        baselines_root=tmp_path / "baselines_a",
    )
    result_b = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        path_b,
        model_ready_root=fixture.model_ready_root,
        baselines_root=tmp_path / "baselines_b",
    )
    assert result_a.manifest.baseline_run_id != result_b.manifest.baseline_run_id


def test_all_candidate_validation_predictions_persisted(tmp_path) -> None:
    fixture, model_result = _build_model_ready(tmp_path)
    config_path = write_baseline_config(tmp_path / "baseline.json")
    result = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=tmp_path / "baselines",
    )
    val_count = result.manifest.validation_record_count
    candidate_count = 3
    assert result.manifest.validation_prediction_count == val_count * candidate_count
    lines = (result.baseline_dir / "predictions_validation.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == val_count * candidate_count


def test_model_consistency_verified(tmp_path) -> None:
    fixture, model_result = _build_model_ready(tmp_path)
    config_path = write_baseline_config(tmp_path / "baseline.json")
    result = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=tmp_path / "baselines",
    )
    verify_baseline_model_consistency(
        result.baseline_dir,
        expected_repository=fixture.repository,
        model_dataset_dir=model_result.model_dataset_dir,
        model_dataset_manifest=model_result.manifest,
        label_map=model_result.label_map,
        trust_model_file=True,
    )


def test_corrupt_manifest_rejected(tmp_path) -> None:
    fixture, model_result = _build_model_ready(tmp_path)
    config_path = write_baseline_config(tmp_path / "baseline.json")
    baselines_root = tmp_path / "baselines"
    result = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=baselines_root,
    )
    manifest_path = result.baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_candidate_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(BaselineCorruptionError):
        validate_baseline_artifact_integrity(
            result.baseline_dir,
            expected_repository=fixture.repository,
            expected_baseline_run_id=result.manifest.baseline_run_id,
        )
