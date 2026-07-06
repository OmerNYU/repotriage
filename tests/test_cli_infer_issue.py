"""Tests for infer-issue CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.cli import main

_CONFIG = Path("configs/inference/pandas-dev__pandas/local-v1.json")


def _artifacts_present() -> bool:
    return _CONFIG.is_file() and (
        Path("data/baselines/github/pandas-dev__pandas").is_dir()
    )


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_cli_infer_issue_success(capsys) -> None:
    code = main(
        [
            "infer-issue",
            "--repo",
            "pandas-dev/pandas",
            "--config",
            str(_CONFIG),
            "--title",
            "BUG: loc indexing returns unexpected result",
            "--body",
            "When using .loc with a list indexer, result dtype is wrong.",
            "--top-k",
            "5",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert '"schema_version": "1"' in output or '"schema_version":"1"' in output
    assert "pandas-dev/pandas" in output
    assert "feature_text_sha256" in output


def test_cli_invalid_repo(capsys, tmp_path: Path) -> None:
    from tests.helpers import write_inference_config

    config_path = write_inference_config(
        tmp_path / "infer.json",
        model_dataset_id="md",
        baseline_run_id="bl",
        threshold_policy_id="tp",
        abstention_policy_id="ap",
        retrieval_run_id="rb",
    )
    code = main(
        [
            "infer-issue",
            "--repo",
            "not-a-repo",
            "--config",
            str(config_path),
            "--title",
            "Title",
        ]
    )
    assert code == 2
    assert capsys.readouterr().err


def test_cli_config_repository_mismatch(capsys) -> None:
    if not _CONFIG.is_file():
        pytest.skip("inference config not present")
    code = main(
        [
            "infer-issue",
            "--repo",
            "other/repo",
            "--config",
            str(_CONFIG),
            "--title",
            "Title",
        ]
    )
    assert code == 2
    assert "does not match" in capsys.readouterr().err
