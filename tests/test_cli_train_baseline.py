"""Tests for train-baseline CLI."""

from __future__ import annotations

from pathlib import Path

from repotriage.cli import main
from tests.helpers import write_baseline_config
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
    config_path = write_baseline_config(tmp_path / "baseline.json")
    return fixture, model_result, config_path


def test_cli_success(tmp_path: Path, capsys) -> None:
    fixture, model_result, config_path = _prepare(tmp_path)
    code = main(
        [
            "train-baseline",
            "--repo",
            "pandas-dev/pandas",
            "--model-dataset-id",
            model_result.manifest.model_dataset_id,
            "--config",
            str(config_path),
            "--model-ready-root",
            str(fixture.model_ready_root),
            "--baselines-root",
            str(tmp_path / "baselines"),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Baseline run ID:" in output
    assert "Baseline-cache hit: no" in output


def test_cli_malformed_model_dataset_id() -> None:
    code = main(
        [
            "train-baseline",
            "--repo",
            "pandas-dev/pandas",
            "--model-dataset-id",
            "bad-id",
            "--config",
            "configs/baselines/pandas-dev__pandas/tfidf-logreg-v1.json",
        ]
    )
    assert code == 2


def test_cli_missing_config_reports_error(tmp_path: Path, capsys) -> None:
    fixture, model_result, _config_path = _prepare(tmp_path)
    code = main(
        [
            "train-baseline",
            "--repo",
            "pandas-dev/pandas",
            "--model-dataset-id",
            model_result.manifest.model_dataset_id,
            "--config",
            str(tmp_path / "missing.json"),
            "--model-ready-root",
            str(fixture.model_ready_root),
            "--baselines-root",
            str(tmp_path / "baselines"),
        ]
    )
    assert code == 1
    assert capsys.readouterr().err
