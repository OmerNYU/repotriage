"""Tests for serve CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.api.settings import ApiSettings
from repotriage.cli import main
from tests.helpers import (
    TEST_ABSTENTION_POLICY_ID,
    TEST_BASELINE_RUN_ID,
    TEST_MODEL_DATASET_ID,
    TEST_RETRIEVAL_RUN_ID,
    TEST_THRESHOLD_POLICY_ID,
)


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


def test_serve_database_url_flag_overrides_default(tmp_path: Path) -> None:
    from argparse import Namespace

    from tests.helpers import write_inference_config

    config_path = write_inference_config(
        tmp_path / "infer.json",
        model_dataset_id=TEST_MODEL_DATASET_ID,
        baseline_run_id=TEST_BASELINE_RUN_ID,
        threshold_policy_id=TEST_THRESHOLD_POLICY_ID,
        abstention_policy_id=TEST_ABSTENTION_POLICY_ID,
        retrieval_run_id=TEST_RETRIEVAL_RUN_ID,
    )
    database_url = f"sqlite:///{tmp_path / 'cli-feedback.db'}"
    args = Namespace(
        config=config_path,
        database_url=database_url,
        baselines_root=Path("data/baselines/github"),
        threshold_policies_root=Path("data/threshold_policies/github"),
        abstention_policies_root=Path("data/abstention_policies/github"),
        retrieval_baselines_root=Path("data/retrieval_baselines/github"),
        model_ready_root=Path("data/model_ready/github"),
    )

    settings = ApiSettings.from_namespace(args)

    assert settings.database_url == database_url
