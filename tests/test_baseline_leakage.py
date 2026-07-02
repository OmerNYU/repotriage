"""Leakage tests for baseline training."""

from __future__ import annotations

from unittest.mock import patch

from repotriage.baseline.builder import train_baseline
from repotriage.baseline.config import load_baseline_config
from repotriage.baseline.reader import load_training_splits
from repotriage.baseline.selector import run_candidate_selection
from repotriage.baseline.trainer import train_candidate
from tests.helpers import write_baseline_config
from tests.test_model_dataset_builder import _setup


def test_vectorizer_and_model_fit_train_only(tmp_path) -> None:
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
    config, _, _, _ = load_baseline_config(config_path)
    splits = load_training_splits(
        model_result.model_dataset_dir,
        expected_repository=fixture.repository,
        expected_model_dataset_id=model_result.manifest.model_dataset_id,
    )
    trained = train_candidate(
        candidate=config.candidates[0],
        splits=splits,
        random_state=config.random_state,
        threshold=config.threshold_policy.threshold,
    )
    val_token = "zz_unique_val_token_for_leakage_test"
    assert val_token not in trained.model.vectorizer.vocabulary_
    splits.validation.texts.append(val_token)
    matrix = trained.model.vectorizer.transform(splits.validation.texts)
    assert matrix[-1, :].nnz == 0


def test_selection_does_not_load_test_split(tmp_path) -> None:
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
    config, _, _, _ = load_baseline_config(config_path)
    splits = load_training_splits(
        model_result.model_dataset_dir,
        expected_repository=fixture.repository,
        expected_model_dataset_id=model_result.manifest.model_dataset_id,
    )

    def _raise_test_load(*_args, **_kwargs):
        raise AssertionError("load_test_split must not be called during selection")

    with patch("repotriage.baseline.builder.load_test_split", side_effect=_raise_test_load):
        selection = run_candidate_selection(
            config=config,
            splits=splits,
            repository=fixture.repository.full_name,
            model_dataset_id=model_result.manifest.model_dataset_id,
            baseline_run_id="test-run-id",
        )
    assert selection.winner_id in {candidate.candidate_id for candidate in config.candidates}


def test_candidate_selection_does_not_use_test_metrics(tmp_path) -> None:
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
    baselines_root = tmp_path / "baselines"
    result = train_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        baselines_root=baselines_root,
    )
    candidate_results = (
        result.baseline_dir / "candidate_results.json"
    ).read_text(encoding="utf-8")
    assert "winner_candidate_id" in candidate_results
    assert result.manifest.test_record_count > 0
