"""Leakage tests for retrieval-baseline building."""

from __future__ import annotations

from unittest.mock import patch

from repotriage.model_dataset.builder import build_model_dataset
from repotriage.retrieval.builder import build_retrieval_baseline
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def test_build_defers_test_split_until_after_validation(tmp_path) -> None:
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
    validation_metrics_completed = {"value": False}

    original_metrics = __import__(
        "repotriage.retrieval.evaluator", fromlist=["compute_retrieval_metrics"]
    ).compute_retrieval_metrics

    def wrapped_metrics(*args, **kwargs):
        result = original_metrics(*args, **kwargs)
        if kwargs.get("split") == "validation":
            validation_metrics_completed["value"] = True
        return result

    def guarded_load(*_args, **_kwargs):
        assert validation_metrics_completed["value"], (
            "test split loaded before validation retrieval metrics"
        )
        return __import__(
            "repotriage.baseline.reader", fromlist=["load_test_split"]
        ).load_test_split(*_args, **_kwargs)

    with patch(
        "repotriage.retrieval.builder.compute_retrieval_metrics",
        side_effect=wrapped_metrics,
    ):
        with patch("repotriage.retrieval.builder.load_test_split", side_effect=guarded_load):
            build_retrieval_baseline(
                fixture.repository,
                model_result.manifest.model_dataset_id,
                config_path,
                model_ready_root=fixture.model_ready_root,
                retrieval_baselines_root=tmp_path / "retrieval_baselines",
            )


def test_neighbors_are_train_only(tmp_path) -> None:
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
    result = build_retrieval_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        retrieval_baselines_root=tmp_path / "retrieval_baselines",
    )
    train_ids = set(result.manifest.train_issue_ids)
    for relative in ("neighbors_validation.jsonl", "neighbors_test.jsonl"):
        path = result.retrieval_dir / relative
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = __import__("json").loads(line)
            for neighbor in payload["neighbors"]:
                assert neighbor["neighbor_issue_id"] in train_ids
