"""Tests for downstream model-dataset artifact integrity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from repotriage.label_policy.builder import validate_label_policy_artifact_integrity
from repotriage.model_dataset.builder import (
    build_model_dataset,
    validate_model_dataset_against_inputs,
    validate_model_dataset_artifact_integrity,
)
from repotriage.model_dataset.config import load_split_config
from repotriage.model_dataset.models import (
    MODEL_DATASET_VERSION,
    MODEL_READY_RECORD_SCHEMA_VERSION,
    OUTPUT_CONTRACT_VERSIONS,
    SPLIT_REPORT_SCHEMA_VERSION,
    ModelDatasetCorruptionError,
    compute_model_dataset_input_sha256,
)
from tests.test_model_dataset_builder import _setup


def test_artifact_integrity_without_source_artifacts(tmp_path: Path) -> None:
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
    manifest, label_map, split_report = validate_model_dataset_artifact_integrity(
        result.model_dataset_dir,
        expected_repository=fixture.repository,
        expected_model_dataset_id=result.manifest.model_dataset_id,
    )
    assert manifest.model_dataset_id == result.manifest.model_dataset_id
    assert label_map.target_count == manifest.target_count
    assert split_report.total_records == manifest.records_written


def test_against_inputs_requires_source_derivation(tmp_path: Path) -> None:
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
    dataset_dir = fixture.processed_root / fixture.repository.slug / fixture.dataset_id
    policy_dir = fixture.policies_root / fixture.repository.slug / fixture.policy_id
    _, policy_document = validate_label_policy_artifact_integrity(
        policy_dir,
        expected_repository=fixture.repository,
        expected_dataset_id=fixture.dataset_id,
        expected_dataset_output_sha256=fixture.processed_manifest.output_sha256,
        expected_policy_id=fixture.policy_id,
    )
    config, config_hash = load_split_config(fixture.split_config_path)
    policy_json_sha256 = hashlib.sha256(
        (policy_dir / "label_policy.json").read_bytes()
    ).hexdigest()
    input_hash = compute_model_dataset_input_sha256(
        model_dataset_version=MODEL_DATASET_VERSION,
        dataset_id=fixture.dataset_id,
        dataset_output_sha256=fixture.processed_manifest.output_sha256,
        policy_id=fixture.policy_id,
        policy_json_sha256=policy_json_sha256,
        text_representation_version=result.manifest.text_representation_version,
        temporal_splitter_version=result.manifest.temporal_splitter_version,
        split_config_schema_version=config.config_schema_version,
        split_config_sha256=config_hash,
    )
    validate_model_dataset_against_inputs(
        result.model_dataset_dir,
        expected_repository=fixture.repository,
        dataset_dir=dataset_dir,
        processed_manifest=fixture.processed_manifest,
        policy_document=policy_document,
        policy_id=fixture.policy_id,
        policy_json_sha256=policy_json_sha256,
        config=config,
        config_hash=config_hash,
        expected_model_dataset_id=result.manifest.model_dataset_id,
        expected_model_dataset_input_sha256=input_hash,
    )


def test_output_contract_versions_documented() -> None:
    from repotriage.model_dataset.models import LABEL_MAP_SCHEMA_VERSION

    assert OUTPUT_CONTRACT_VERSIONS == (
        MODEL_READY_RECORD_SCHEMA_VERSION,
        LABEL_MAP_SCHEMA_VERSION,
        SPLIT_REPORT_SCHEMA_VERSION,
    )


def _tamper_first_record(
    model_dataset_dir: Path, *, mutate
) -> None:
    manifest_path = model_dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = model_dataset_dir / manifest["records_file"]
    lines = records_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    mutate(record)
    lines[0] = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    records_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    records_path.write_bytes(records_bytes)
    manifest["records_sha256"] = hashlib.sha256(records_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_feature_text_tamper_rejected(tmp_path: Path) -> None:
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

    def mutate(record: dict) -> None:
        record["feature_text"] = "tampered"

    _tamper_first_record(result.model_dataset_dir, mutate=mutate)
    with pytest.raises(ModelDatasetCorruptionError, match="feature_text"):
        validate_model_dataset_artifact_integrity(
            result.model_dataset_dir,
            expected_repository=fixture.repository,
            expected_model_dataset_id=result.manifest.model_dataset_id,
        )


def test_stale_feature_text_after_title_tamper_rejected(tmp_path: Path) -> None:
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

    def mutate(record: dict) -> None:
        record["title"] = "Tampered title"

    _tamper_first_record(result.model_dataset_dir, mutate=mutate)
    with pytest.raises(ModelDatasetCorruptionError, match="feature_text"):
        validate_model_dataset_artifact_integrity(
            result.model_dataset_dir,
            expected_repository=fixture.repository,
            expected_model_dataset_id=result.manifest.model_dataset_id,
        )


def _tamper_split_report_warnings(model_dataset_dir: Path, *, warnings: list[dict]) -> None:
    manifest_path = model_dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = model_dataset_dir / manifest["split_report_json_file"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["warnings"] = warnings
    report["support_validation"]["warnings"] = warnings
    report_bytes = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    report_path.write_bytes(report_bytes)
    manifest["split_report_json_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_noncanonical_warning_order_rejected(tmp_path: Path) -> None:
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
    report_path = result.model_dataset_dir / "split_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reversed_warnings = list(reversed(report["warnings"]))
    if reversed_warnings == report["warnings"]:
        reversed_warnings = report["warnings"][1:] + report["warnings"][:1]
    _tamper_split_report_warnings(
        result.model_dataset_dir,
        warnings=reversed_warnings,
    )
    with pytest.raises(ModelDatasetCorruptionError, match="canonical order"):
        validate_model_dataset_artifact_integrity(
            result.model_dataset_dir,
            expected_repository=fixture.repository,
            expected_model_dataset_id=result.manifest.model_dataset_id,
        )


def test_boolean_target_vector_rejected_by_artifact_validator(tmp_path: Path) -> None:
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

    def mutate(record: dict) -> None:
        record["target_vector"] = [True] + [0] * (len(record["target_vector"]) - 1)

    _tamper_first_record(result.model_dataset_dir, mutate=mutate)
    with pytest.raises(ModelDatasetCorruptionError):
        validate_model_dataset_artifact_integrity(
            result.model_dataset_dir,
            expected_repository=fixture.repository,
            expected_model_dataset_id=result.manifest.model_dataset_id,
        )
