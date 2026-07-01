"""Tests for build-model-dataset CLI."""

from __future__ import annotations

from pathlib import Path

from repotriage.cli import main
from tests.test_model_dataset_builder import _setup


def test_cli_success(tmp_path: Path, capsys) -> None:
    fixture = _setup(tmp_path)
    code = main(
        [
            "build-model-dataset",
            "--repo",
            "pandas-dev/pandas",
            "--dataset-id",
            fixture.dataset_id,
            "--policy-id",
            fixture.policy_id,
            "--config",
            str(fixture.split_config_path),
            "--processed-root",
            str(fixture.processed_root),
            "--policies-root",
            str(fixture.policies_root),
            "--model-ready-root",
            str(fixture.model_ready_root),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Model-dataset ID:" in output
    assert "Model-dataset-cache hit: no" in output


def test_cli_malformed_dataset_id() -> None:
    code = main(
        [
            "build-model-dataset",
            "--repo",
            "pandas-dev/pandas",
            "--dataset-id",
            "bad-id",
            "--policy-id",
            "20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37",
            "--config",
            "configs/model_datasets/pandas-dev__pandas/temporal-v1.json",
        ]
    )
    assert code == 2


def test_cli_malformed_policy_id() -> None:
    code = main(
        [
            "build-model-dataset",
            "--repo",
            "pandas-dev/pandas",
            "--dataset-id",
            "20260628T161306010651Z-n1-074402d21505",
            "--policy-id",
            "bad-policy",
            "--config",
            "configs/model_datasets/pandas-dev__pandas/temporal-v1.json",
        ]
    )
    assert code == 2


def test_cli_missing_config_reports_error(tmp_path: Path, capsys) -> None:
    fixture = _setup(tmp_path)
    code = main(
        [
            "build-model-dataset",
            "--repo",
            "pandas-dev/pandas",
            "--dataset-id",
            fixture.dataset_id,
            "--policy-id",
            fixture.policy_id,
            "--config",
            str(tmp_path / "missing.json"),
            "--processed-root",
            str(fixture.processed_root),
            "--policies-root",
            str(fixture.policies_root),
            "--model-ready-root",
            str(fixture.model_ready_root),
        ]
    )
    assert code == 1
    assert "configuration" in capsys.readouterr().err.lower()
