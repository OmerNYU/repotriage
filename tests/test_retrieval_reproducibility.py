"""Two-root reproducibility tests for retrieval-baseline artifacts."""

from __future__ import annotations

from pathlib import Path

from repotriage.model_dataset.builder import build_model_dataset
from repotriage.retrieval.builder import build_retrieval_baseline
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def test_same_inputs_different_roots_same_identity(tmp_path: Path) -> None:
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

    result_a = build_retrieval_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        retrieval_baselines_root=tmp_path / "retrieval_a",
    )
    result_b = build_retrieval_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        retrieval_baselines_root=tmp_path / "retrieval_b",
    )

    assert result_a.manifest.retrieval_run_id == result_b.manifest.retrieval_run_id
    assert (
        result_a.validation_metrics.recall_at_5
        == result_b.validation_metrics.recall_at_5
    )
    assert result_a.test_metrics.recall_at_10 == result_b.test_metrics.recall_at_10
