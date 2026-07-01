"""Real-data acceptance tests for the pandas model-ready dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.github.models import RepositoryRef
from repotriage.model_dataset.builder import build_model_dataset

_DATASET_ID = "20260628T161306010651Z-n1-074402d21505"
_POLICY_ID = "20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37"
_MODEL_DATASET_ID = "20260628T161306010651Z-n1-074402d21505-md1-14a9768bded7"
_CONFIG = Path("configs/model_datasets/pandas-dev__pandas/temporal-v1.json")
_PROCESSED = Path("data/processed/github")
_POLICIES = Path("data/policies/github")
_MODEL_READY = Path("data/model_ready/github")
_EXISTING_ARTIFACT = (
    _MODEL_READY / "pandas-dev__pandas" / _MODEL_DATASET_ID
)


def _artifacts_present() -> bool:
    return (
        (_PROCESSED / "pandas-dev__pandas" / _DATASET_ID).is_dir()
        and (_POLICIES / "pandas-dev__pandas" / _POLICY_ID).is_dir()
        and _CONFIG.is_file()
    )


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_real_data_acceptance() -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    result = build_model_dataset(
        repository,
        _DATASET_ID,
        _POLICY_ID,
        _CONFIG,
        processed_root=_PROCESSED,
        policies_root=_POLICIES,
        model_ready_root=_MODEL_READY,
    )

    manifest = result.manifest
    report = result.split_report
    gts = report.global_target_statistics

    assert manifest.model_dataset_id == _MODEL_DATASET_ID
    assert manifest.records_written == 1119
    assert manifest.target_count == 15
    assert gts.all_zero_target_count == 154
    assert gts.positive_assignments == 1416
    assert gts.issues_with_included_target == 965
    assert gts.issues_without_included_target == 154
    assert gts.target_coverage_fraction == pytest.approx(965 / 1119)

    assert report.splits["train"].issue_count == 840
    assert report.splits["validation"].issue_count == 152
    assert report.splits["test"].issue_count == 127
    assert (
        report.splits["train"].issue_count
        + report.splits["validation"].issue_count
        + report.splits["test"].issue_count
        == 1119
    )

    assert report.splits["train"].all_zero_target_count == 105
    assert report.splits["validation"].all_zero_target_count == 25
    assert report.splits["test"].all_zero_target_count == 24

    bug = report.splits["train"].positives_per_label["Bug"]
    assert bug == 421
    assert report.splits["validation"].positives_per_label["Bug"] == 55
    assert report.splits["test"].positives_per_label["Bug"] == 60

    assert result.cache_hit is True

    slug_dir = _MODEL_READY / repository.slug
    staging = [p for p in slug_dir.iterdir() if p.name.startswith(".") and "staging" in p.name]
    assert staging == []


@pytest.mark.integration
@pytest.mark.skipif(not _artifacts_present(), reason="canonical local artifacts not present")
def test_fresh_build_matches_existing_deterministic_bytes(tmp_path: Path) -> None:
    """Fresh build must byte-match existing artifact for all deterministic output files."""
    import hashlib

    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    fresh_root = tmp_path / "model_ready"
    result = build_model_dataset(
        repository,
        _DATASET_ID,
        _POLICY_ID,
        _CONFIG,
        processed_root=_PROCESSED,
        policies_root=_POLICIES,
        model_ready_root=fresh_root,
    )
    assert result.manifest.model_dataset_id == _MODEL_DATASET_ID
    fresh_dir = result.model_dataset_dir
    for name in (
        "records.jsonl",
        "label_map.json",
        "split_report.json",
        "split_report.md",
    ):
        existing_bytes = (_EXISTING_ARTIFACT / name).read_bytes()
        fresh_bytes = (fresh_dir / name).read_bytes()
        assert existing_bytes == fresh_bytes, name
        assert hashlib.sha256(existing_bytes).hexdigest() == hashlib.sha256(fresh_bytes).hexdigest()
