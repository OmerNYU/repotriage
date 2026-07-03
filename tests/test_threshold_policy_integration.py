"""Real-data acceptance tests for the pandas threshold-policy artifact."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from repotriage.baseline.builder import validate_baseline_artifact_integrity
from repotriage.github.models import RepositoryRef
from repotriage.threshold_policy.builder import (
    build_threshold_policy,
    validate_threshold_policy_against_baseline,
)
from repotriage.threshold_policy.models import ComparisonDocument, SweepValidationDocument

_BASELINE_RUN_ID = (
    "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7-bl4-46227a0ec602"
)
_CONFIG = Path("configs/threshold_policies/pandas-dev__pandas/global-v1.json")
_BASELINES = Path("data/baselines/github")
_BASELINE_DIR = _BASELINES / "pandas-dev__pandas" / _BASELINE_RUN_ID


def _artifacts_present() -> bool:
    return _BASELINE_DIR.is_dir() and _CONFIG.is_file()


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_real_data_threshold_policy_acceptance(tmp_path: Path) -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    policies_root = tmp_path / "threshold_policies"
    selection_completed = {"value": False}

    original_pipeline = __import__(
        "repotriage.threshold_policy.builder", fromlist=["_run_validation_only_pipeline"]
    )._run_validation_only_pipeline

    def wrapped_pipeline(*args, **kwargs):
        result = original_pipeline(*args, **kwargs)
        selection_completed["value"] = True
        return result

    def guarded_load(*_args, **_kwargs):
        assert selection_completed["value"], "test scores loaded before selection"
        return __import__(
            "repotriage.threshold_policy.reader", fromlist=["load_test_scores"]
        ).load_test_scores(*_args, **_kwargs)

    with patch(
        "repotriage.threshold_policy.builder._run_validation_only_pipeline",
        side_effect=wrapped_pipeline,
    ):
        with patch(
            "repotriage.threshold_policy.builder.load_test_scores",
            side_effect=guarded_load,
        ):
            result = build_threshold_policy(
                repository,
                _CONFIG,
                baselines_root=_BASELINES,
                threshold_policies_root=policies_root,
            )

    assert result.cache_hit is False
    assert "-tp1-" in result.manifest.policy_id
    assert result.manifest.validation_record_count == 152
    assert result.manifest.test_record_count == 127
    assert result.manifest.target_count == 15

    sweep = SweepValidationDocument.model_validate_json(
        (result.policy_dir / "sweep_validation.json").read_text(encoding="utf-8")
    )
    assert len(sweep.rows) == 91
    assert result.selected_threshold_basis_points in {
        row.threshold_basis_points for row in sweep.rows
    }

    comparison = ComparisonDocument.model_validate_json(
        (result.policy_dir / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison.reference_threshold_basis_points == 50
    assert comparison.validation.reference.macro_f1 == pytest.approx(0.4745023527490588, abs=1e-12)
    assert comparison.test.reference.macro_f1 == pytest.approx(0.49809593175817085, abs=1e-12)

    selected_row = next(
        row
        for row in sweep.rows
        if row.threshold_basis_points == result.selected_threshold_basis_points
    )
    metrics_validation = json.loads(
        (result.policy_dir / "metrics_validation.json").read_text(encoding="utf-8")
    )
    assert metrics_validation["aggregate"]["macro_f1"] == pytest.approx(
        selected_row.metrics.aggregate.macro_f1,
        abs=1e-12,
    )

    validate_baseline_artifact_integrity(
        _BASELINE_DIR,
        expected_repository=repository,
        expected_baseline_run_id=_BASELINE_RUN_ID,
    )
    validate_threshold_policy_against_baseline(
        result.policy_dir,
        _BASELINE_DIR,
        expected_repository=repository,
    )

    second = build_threshold_policy(
        repository,
        _CONFIG,
        baselines_root=_BASELINES,
        threshold_policies_root=policies_root,
    )
    assert second.cache_hit is True

    slug_dir = policies_root / repository.slug
    staging = [p for p in slug_dir.iterdir() if p.name.startswith(".") and "staging" in p.name]
    assert staging == []
