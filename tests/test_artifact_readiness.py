"""Tests for inference artifact readiness checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repotriage.cli import main
from repotriage.github.models import RepositoryRef
from repotriage.inference.readiness import (
    ArtifactRoots,
    ReadinessMode,
    check_inference_artifacts,
    format_readiness_report,
)
from repotriage.model_dataset.builder import build_model_dataset
from tests.helpers import TEST_BASELINE_RUN_ID, TEST_MODEL_DATASET_ID, write_inference_config
from tests.test_model_dataset_builder import _setup

_CONFIG = Path("configs/inference/pandas-dev__pandas/local-v1.json")
_REPOSITORY = "pandas-dev/pandas"
_SLUG = "pandas-dev__pandas"


def _write_config(path: Path) -> Path:
    return write_inference_config(
        path,
        repository=_REPOSITORY,
        model_dataset_id=TEST_MODEL_DATASET_ID,
        baseline_run_id=TEST_BASELINE_RUN_ID,
        threshold_policy_id=f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458",
        abstention_policy_id=f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458-ap1-9c3c140e7ccb",
        retrieval_run_id=f"{TEST_MODEL_DATASET_ID}-rb1-deb29b6da4eb",
    )


def _artifact_roots(tmp_path: Path) -> ArtifactRoots:
    return ArtifactRoots(
        model_ready_root=tmp_path / "model_ready",
        baselines_root=tmp_path / "baselines",
        threshold_policies_root=tmp_path / "threshold_policies",
        abstention_policies_root=tmp_path / "abstention_policies",
        retrieval_baselines_root=tmp_path / "retrieval_baselines",
    )


def _touch_manifest(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")


def _canonical_artifacts_present() -> bool:
    if not _CONFIG.is_file():
        return False
    paths = (
        Path("data/model_ready/github") / _SLUG / TEST_MODEL_DATASET_ID,
        Path("data/baselines/github") / _SLUG / TEST_BASELINE_RUN_ID,
        Path("data/threshold_policies/github")
        / _SLUG
        / f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458",
        Path("data/abstention_policies/github")
        / _SLUG
        / f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458-ap1-9c3c140e7ccb",
        Path("data/retrieval_baselines/github")
        / _SLUG
        / f"{TEST_MODEL_DATASET_ID}-rb1-deb29b6da4eb",
    )
    return all(path.is_dir() for path in paths)


def test_check_missing_all_artifacts(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "infer.json")
    roots = _artifact_roots(tmp_path)

    report = check_inference_artifacts(config_path, roots=roots)

    assert report.ready is False
    assert all(result.status == "missing" for result in report.results)
    rendered = format_readiness_report(report)
    assert "[MISSING]" in rendered
    assert "Not ready." in rendered


def test_check_missing_single_artifact(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "infer.json")
    roots = _artifact_roots(tmp_path)
    repository = RepositoryRef(owner="pandas-dev", name="pandas")

    _touch_manifest(roots.model_ready_root / repository.slug / TEST_MODEL_DATASET_ID)
    _touch_manifest(
        roots.threshold_policies_root / repository.slug / f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458"
    )
    _touch_manifest(
        roots.abstention_policies_root
        / repository.slug
        / f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458-ap1-9c3c140e7ccb"
    )
    _touch_manifest(
        roots.retrieval_baselines_root
        / repository.slug
        / f"{TEST_MODEL_DATASET_ID}-rb1-deb29b6da4eb"
    )

    report = check_inference_artifacts(config_path, roots=roots)

    assert report.ready is False
    statuses = {result.name: result.status for result in report.results}
    assert statuses["baseline classifier"] == "missing"
    assert statuses["model-ready dataset"] == "ok"


def test_check_presence_ok_with_manifest_only(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "infer.json")
    roots = _artifact_roots(tmp_path)
    repository = RepositoryRef(owner="pandas-dev", name="pandas")

    _touch_manifest(roots.model_ready_root / repository.slug / TEST_MODEL_DATASET_ID)
    _touch_manifest(roots.baselines_root / repository.slug / TEST_BASELINE_RUN_ID)
    _touch_manifest(
        roots.threshold_policies_root / repository.slug / f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458"
    )
    _touch_manifest(
        roots.abstention_policies_root
        / repository.slug
        / f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458-ap1-9c3c140e7ccb"
    )
    _touch_manifest(
        roots.retrieval_baselines_root
        / repository.slug
        / f"{TEST_MODEL_DATASET_ID}-rb1-deb29b6da4eb"
    )

    report = check_inference_artifacts(config_path, mode=ReadinessMode.PRESENCE, roots=roots)

    assert report.ready is True
    assert all(result.status == "ok" for result in report.results)


def test_check_integrity_catches_hash_mismatch(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    records_path = result.model_dataset_dir / "records.jsonl"
    records_path.write_text(records_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    config_path = write_inference_config(
        tmp_path / "infer.json",
        repository=fixture.repository.full_name,
        model_dataset_id=result.manifest.model_dataset_id,
        baseline_run_id=f"{result.manifest.model_dataset_id}-bl4-000000000000",
        threshold_policy_id=f"{result.manifest.model_dataset_id}-bl4-000000000000-tp1-000000000000",
        abstention_policy_id=(
            f"{result.manifest.model_dataset_id}-bl4-000000000000-tp1-000000000000-ap1-000000000000"
        ),
        retrieval_run_id=f"{result.manifest.model_dataset_id}-rb1-000000000000",
    )
    roots = ArtifactRoots(model_ready_root=fixture.model_ready_root)

    report = check_inference_artifacts(
        config_path,
        mode=ReadinessMode.INTEGRITY,
        roots=roots,
    )

    assert report.ready is False
    model_result = report.results[0]
    assert model_result.status == "invalid"
    assert model_result.detail is not None


def test_next_command_includes_config_ids(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "infer.json")
    roots = _artifact_roots(tmp_path)

    report = check_inference_artifacts(config_path, roots=roots)
    baseline = next(result for result in report.results if result.name == "baseline classifier")

    assert baseline.next_command is not None
    assert "repotriage train-baseline" in baseline.next_command
    assert TEST_MODEL_DATASET_ID in baseline.next_command


def test_cli_check_artifacts_exit_codes(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path / "infer.json")
    roots = _artifact_roots(tmp_path)

    code = main(
        [
            "check-artifacts",
            "--config",
            str(config_path),
            "--model-ready-root",
            str(roots.model_ready_root),
            "--baselines-root",
            str(roots.baselines_root),
            "--threshold-policies-root",
            str(roots.threshold_policies_root),
            "--abstention-policies-root",
            str(roots.abstention_policies_root),
            "--retrieval-baselines-root",
            str(roots.retrieval_baselines_root),
        ]
    )
    assert code == 1
    assert "[MISSING]" in capsys.readouterr().out

    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    for root, artifact_id in (
        (roots.model_ready_root, TEST_MODEL_DATASET_ID),
        (roots.baselines_root, TEST_BASELINE_RUN_ID),
        (roots.threshold_policies_root, f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458"),
        (
            roots.abstention_policies_root,
            f"{TEST_BASELINE_RUN_ID}-tp1-ccaab0996458-ap1-9c3c140e7ccb",
        ),
        (roots.retrieval_baselines_root, f"{TEST_MODEL_DATASET_ID}-rb1-deb29b6da4eb"),
    ):
        _touch_manifest(root / repository.slug / artifact_id)

    code = main(
        [
            "check-artifacts",
            "--config",
            str(config_path),
            "--model-ready-root",
            str(roots.model_ready_root),
            "--baselines-root",
            str(roots.baselines_root),
            "--threshold-policies-root",
            str(roots.threshold_policies_root),
            "--abstention-policies-root",
            str(roots.abstention_policies_root),
            "--retrieval-baselines-root",
            str(roots.retrieval_baselines_root),
        ]
    )
    assert code == 0
    assert "Ready for:" in capsys.readouterr().out


def test_cli_check_artifacts_json_output(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path / "infer.json")
    roots = _artifact_roots(tmp_path)

    code = main(
        [
            "check-artifacts",
            "--config",
            str(config_path),
            "--json",
            "--model-ready-root",
            str(roots.model_ready_root),
            "--baselines-root",
            str(roots.baselines_root),
            "--threshold-policies-root",
            str(roots.threshold_policies_root),
            "--abstention-policies-root",
            str(roots.abstention_policies_root),
            "--retrieval-baselines-root",
            str(roots.retrieval_baselines_root),
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["mode"] == "presence"
    assert len(payload["artifacts"]) == 5


@pytest.mark.integration
@pytest.mark.skipif(
    not _canonical_artifacts_present(),
    reason="canonical local artifacts not present",
)
def test_check_strict_with_canonical_artifacts() -> None:
    report = check_inference_artifacts(_CONFIG, mode=ReadinessMode.STRICT)
    assert report.ready is True
    assert report.strict_error is None
