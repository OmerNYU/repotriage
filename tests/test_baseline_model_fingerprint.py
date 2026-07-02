"""Tests for model semantic fingerprinting (bl4 contract)."""

from __future__ import annotations

import io
import json

import joblib
import numpy as np
import pytest

from repotriage.baseline.builder import train_baseline, verify_baseline_model_consistency
from repotriage.baseline.config import load_baseline_config
from repotriage.baseline.models import BaselineCorruptionError
from repotriage.baseline.models_ml import model_semantic_sha256, train_model
from tests.helpers import write_baseline_config
from tests.test_model_dataset_builder import _setup


def _train_small_model(tmp_path):
    config, _, _, _ = load_baseline_config(write_baseline_config(tmp_path / "baseline.json"))
    candidate = config.candidates[0]
    labels = ["Bug", "Docs"]
    train_texts = [
        "bug crash",
        "documentation fix",
        "another bug",
        "more docs",
    ]
    train_targets = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int8,
    )
    model, _report = train_model(
        candidate=candidate,
        labels=labels,
        train_texts=train_texts,
        train_targets=train_targets,
        random_state=42,
        threshold=0.5,
    )
    return model


def test_identical_models_produce_identical_fingerprint(tmp_path) -> None:
    model_a = _train_small_model(tmp_path)
    model_b = _train_small_model(tmp_path)
    assert model_semantic_sha256(model_a) == model_semantic_sha256(model_b)


def test_changing_vocabulary_index_changes_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    first_term = next(iter(model.vectorizer.vocabulary_))
    model.vectorizer.vocabulary_[first_term] = 99999
    assert model_semantic_sha256(model) != baseline


def test_changing_idf_changes_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    model.vectorizer.idf_[0] = model.vectorizer.idf_[0] + 0.01
    assert model_semantic_sha256(model) != baseline


def test_changing_coefficient_changes_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    model.estimators[0].coef_[0, 0] += 0.01
    assert model_semantic_sha256(model) != baseline


def test_changing_label_order_changes_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    model.labels = list(reversed(model.labels))
    assert model_semantic_sha256(model) != baseline


def test_changing_threshold_changes_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    model.threshold = 0.25
    assert model_semantic_sha256(model) != baseline


def test_mutating_stop_words_id_does_not_change_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    model.vectorizer._stop_words_id = 123456789
    assert model_semantic_sha256(model) == baseline


def test_joblib_round_trip_preserves_semantic_fingerprint(tmp_path) -> None:
    model = _train_small_model(tmp_path)
    baseline = model_semantic_sha256(model)
    buffer_a = io.BytesIO()
    buffer_b = io.BytesIO()
    joblib.dump(model.to_bundle(), buffer_a)
    joblib.dump(model.to_bundle(), buffer_b)
    loaded_a = joblib.load(io.BytesIO(buffer_a.getvalue()))
    loaded_b = joblib.load(io.BytesIO(buffer_b.getvalue()))
    from repotriage.baseline.models_ml import load_model_from_bundle

    assert model_semantic_sha256(load_model_from_bundle(loaded_a)) == baseline
    assert model_semantic_sha256(load_model_from_bundle(loaded_b)) == baseline


def test_tampered_model_fails_trusted_verification(tmp_path) -> None:
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

    bundle = joblib.load(result.baseline_dir / "model.joblib")
    bundle["estimators"][0].coef_[0, 0] += 0.01
    joblib.dump(bundle, result.baseline_dir / "model.joblib")
    with pytest.raises(BaselineCorruptionError):
        verify_baseline_model_consistency(
            result.baseline_dir,
            expected_repository=fixture.repository,
            model_dataset_dir=model_result.model_dataset_dir,
            model_dataset_manifest=model_result.manifest,
            label_map=model_result.label_map,
            trust_model_file=True,
        )


def test_cross_root_builds_share_semantic_hash_but_model_bytes_may_differ(tmp_path) -> None:
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
    config_path = write_baseline_config(tmp_path / "baseline.json")
    root_a = tmp_path / "baselines_a"
    root_b = tmp_path / "baselines_b"
    result_a = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=root_a,
    )
    result_b = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=root_b,
    )

    manifest_a = json.loads((result_a.baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((result_b.baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_a.pop("built_at", None)
    manifest_b.pop("built_at", None)
    model_sha_a = manifest_a.pop("model_sha256")
    model_sha_b = manifest_b.pop("model_sha256")
    assert manifest_a == manifest_b
    assert result_a.manifest.model_semantic_sha256 == result_b.manifest.model_semantic_sha256

    model_bytes_a = (result_a.baseline_dir / "model.joblib").read_bytes()
    model_bytes_b = (result_b.baseline_dir / "model.joblib").read_bytes()
    if model_sha_a != model_sha_b:
        assert model_bytes_a != model_bytes_b
    else:
        assert model_bytes_a == model_bytes_b

    for filename in (
        "config.json",
        "candidate_results.json",
        "metrics_test.json",
        "metrics.md",
        "predictions_validation.jsonl",
        "predictions_test.jsonl",
        "feature_summary.json",
    ):
        assert (result_a.baseline_dir / filename).read_bytes() == (
            result_b.baseline_dir / filename
        ).read_bytes()
