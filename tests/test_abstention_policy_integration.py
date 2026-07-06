"""Real-data acceptance tests for the pandas abstention-policy artifact."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from repotriage.abstention_policy.builder import (
    build_abstention_policy,
    validate_abstention_policy_against_inputs,
)
from repotriage.abstention_policy.models import SweepValidationDocument
from repotriage.github.models import RepositoryRef

_THRESHOLD_POLICY_ID = (
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602-tp1-ccaab0996458"
)
_CONFIG = Path("configs/abstention_policies/pandas-dev__pandas/issue-confidence-v1.json")
_BASELINES = Path("data/baselines/github")
_THRESHOLD_POLICIES = Path("data/threshold_policies/github")
_THRESHOLD_POLICY_DIR = _THRESHOLD_POLICIES / "pandas-dev__pandas" / _THRESHOLD_POLICY_ID
_BASELINE_DIR = _BASELINES / "pandas-dev__pandas" / (
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602"
)


def _artifacts_present() -> bool:
    return _THRESHOLD_POLICY_DIR.is_dir() and _BASELINE_DIR.is_dir() and _CONFIG.is_file()


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_real_data_abstention_policy_acceptance(tmp_path: Path) -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    abstention_root = tmp_path / "abstention_policies"
    selection_completed = {"value": False}

    original_pipeline = __import__(
        "repotriage.abstention_policy.builder", fromlist=["_run_validation_only_pipeline"]
    )._run_validation_only_pipeline

    def wrapped_pipeline(*args, **kwargs):
        result = original_pipeline(*args, **kwargs)
        selection_completed["value"] = True
        return result

    def guarded_load(*_args, **_kwargs):
        assert selection_completed["value"], "test scores loaded before selection"
        return __import__(
            "repotriage.abstention_policy.reader", fromlist=["load_test_scores"]
        ).load_test_scores(*_args, **_kwargs)

    with patch(
        "repotriage.abstention_policy.builder._run_validation_only_pipeline",
        side_effect=wrapped_pipeline,
    ):
        with patch(
            "repotriage.abstention_policy.builder.load_test_scores",
            side_effect=guarded_load,
        ):
            result = build_abstention_policy(
                repository,
                _CONFIG,
                threshold_policy_id=_THRESHOLD_POLICY_ID,
                baselines_root=_BASELINES,
                threshold_policies_root=_THRESHOLD_POLICIES,
                abstention_policies_root=abstention_root,
            )

    assert result.cache_hit is False
    assert "-ap1-" in result.manifest.policy_id
    assert result.classification_threshold_basis_points == 39
    assert result.manifest.validation_record_count == 152
    assert result.manifest.test_record_count == 127
    assert result.manifest.target_count == 15

    sweep = SweepValidationDocument.model_validate_json(
        (result.policy_dir / "sweep_validation.json").read_text(encoding="utf-8")
    )
    assert sweep.abstention_grid.start_basis_points == 39
    assert len(sweep.rows) == 57
    assert result.selected_abstention_basis_points in {
        row.abstention_basis_points for row in sweep.rows
    }
    assert result.selected_abstention_basis_points == 84

    selected_row = next(
        row
        for row in sweep.rows
        if row.abstention_basis_points == result.selected_abstention_basis_points
    )
    metrics_validation = json.loads(
        (result.policy_dir / "metrics_validation.json").read_text(encoding="utf-8")
    )
    assert metrics_validation["handled_metrics"]["subset_accuracy"] == pytest.approx(
        selected_row.handled_metrics.subset_accuracy,
        abs=1e-12,
    )

    validate_abstention_policy_against_inputs(
        result.policy_dir,
        _BASELINE_DIR,
        _THRESHOLD_POLICY_DIR,
        expected_repository=repository,
    )

    second = build_abstention_policy(
        repository,
        _CONFIG,
        threshold_policy_id=_THRESHOLD_POLICY_ID,
        baselines_root=_BASELINES,
        threshold_policies_root=_THRESHOLD_POLICIES,
        abstention_policies_root=abstention_root,
    )
    assert second.cache_hit is True
