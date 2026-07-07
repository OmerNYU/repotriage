"""Tests for serve CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.cli import main


def test_serve_invalid_config_returns_2(capsys, tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    code = main(["serve", "--config", str(missing)])
    assert code == 2
    assert capsys.readouterr().err


def test_serve_invalid_port_returns_2(tmp_path: Path) -> None:
    from tests.helpers import write_inference_config

    config_path = write_inference_config(
        tmp_path / "infer.json",
        model_dataset_id="md",
        baseline_run_id="bl",
        threshold_policy_id="tp",
        abstention_policy_id="ap",
        retrieval_run_id="rb",
    )
    code = main(["serve", "--config", str(config_path), "--port", "0"])
    assert code == 2


@pytest.mark.parametrize("port", ["70000", "-1"])
def test_serve_out_of_range_port_returns_2(tmp_path: Path, port: str) -> None:
    from tests.helpers import write_inference_config

    config_path = write_inference_config(
        tmp_path / "infer.json",
        model_dataset_id="md",
        baseline_run_id="bl",
        threshold_policy_id="tp",
        abstention_policy_id="ap",
        retrieval_run_id="rb",
    )
    code = main(["serve", "--config", str(config_path), "--port", port])
    assert code == 2
