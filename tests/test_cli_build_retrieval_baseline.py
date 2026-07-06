"""Tests for build-retrieval-baseline CLI."""

from __future__ import annotations

from pathlib import Path

from repotriage.cli import main
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def _prepare(tmp_path: Path):
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
    config_path = write_retrieval_baseline_config(tmp_path / "retrieval.json", min_df=1)
    return fixture, model_result, config_path


def test_cli_success(tmp_path: Path, capsys) -> None:
    fixture, model_result, config_path = _prepare(tmp_path)
    code = main(
        [
            "build-retrieval-baseline",
            "--repo",
            "pandas-dev/pandas",
            "--model-dataset-id",
            model_result.manifest.model_dataset_id,
            "--config",
            str(config_path),
            "--model-ready-root",
            str(fixture.model_ready_root),
            "--retrieval-baselines-root",
            str(tmp_path / "retrieval_baselines"),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "retrieval_run_id:" in output
    assert "cache_hit: false" in output


def test_cli_invalid_model_dataset_id(tmp_path: Path, capsys) -> None:
    fixture, _model_result, config_path = _prepare(tmp_path)
    code = main(
        [
            "build-retrieval-baseline",
            "--repo",
            "pandas-dev/pandas",
            "--model-dataset-id",
            "not-a-valid-id",
            "--config",
            str(config_path),
            "--model-ready-root",
            str(fixture.model_ready_root),
            "--retrieval-baselines-root",
            str(tmp_path / "retrieval_baselines"),
        ]
    )
    assert code == 2
    assert "Invalid model-dataset id" in capsys.readouterr().err


def test_cli_cache_hit_reported(tmp_path: Path, capsys) -> None:
    fixture, model_result, config_path = _prepare(tmp_path)
    args = [
        "build-retrieval-baseline",
        "--repo",
        "pandas-dev/pandas",
        "--model-dataset-id",
        model_result.manifest.model_dataset_id,
        "--config",
        str(config_path),
        "--model-ready-root",
        str(fixture.model_ready_root),
        "--retrieval-baselines-root",
        str(tmp_path / "retrieval_baselines"),
    ]
    assert main(args) == 0
    capsys.readouterr()
    assert main(args) == 0
    assert "cache_hit: true" in capsys.readouterr().out
