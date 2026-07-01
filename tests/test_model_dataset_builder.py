"""Tests for model-dataset orchestration, publication, determinism, and immutability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repotriage.audit.builder import audit_dataset
from repotriage.dataset.models import ProcessedManifest
from repotriage.github.models import RepositoryRef
from repotriage.label_policy.builder import (
    build_label_policy,
    validate_label_policy_artifact_integrity,
)
from repotriage.model_dataset.builder import (
    build_model_dataset,
    publish_model_dataset,
    validate_model_dataset_against_inputs,
)
from repotriage.model_dataset.config import load_split_config
from repotriage.model_dataset.models import (
    MODEL_DATASET_VERSION,
    TEMPORAL_SPLITTER_VERSION,
    TEXT_REPRESENTATION_VERSION,
    ModelDatasetBuildError,
    ModelDatasetCorruptionError,
    ModelDatasetInputError,
    ModelDatasetSplitSupportError,
    compute_model_dataset_id,
    compute_model_dataset_input_sha256,
)
from tests.helpers import (
    make_normalized_issue,
    write_label_policy_config,
    write_processed_dataset,
    write_temporal_split_config,
)

_CRITERIA = {
    "min_total_support": 2,
    "min_active_months": 2,
    "min_recent_support": 1,
    "recent_window_months": 2,
}


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def _issues() -> list:
    return [
        make_normalized_issue(1, labels=["Bug"], created_at=_at(2025, 1, 5)),
        make_normalized_issue(2, labels=["Bug", "Docs"], created_at=_at(2025, 5, 5)),
        make_normalized_issue(3, labels=["Docs"], created_at=_at(2025, 6, 5)),
        make_normalized_issue(4, labels=["Groupby"], created_at=_at(2025, 3, 5)),
        make_normalized_issue(5, labels=["Needs Triage"], created_at=_at(2025, 4, 5)),
        make_normalized_issue(6, labels=["Misc"], created_at=_at(2025, 2, 5)),
        make_normalized_issue(7, labels=["Bug"], created_at=_at(2026, 2, 10)),
        make_normalized_issue(8, labels=["Docs"], created_at=_at(2026, 3, 15)),
        make_normalized_issue(9, labels=["Bug", "Docs"], created_at=_at(2026, 4, 20)),
        make_normalized_issue(10, labels=["Bug"], created_at=_at(2026, 5, 25)),
    ]


def _labels() -> list[dict[str, object]]:
    return [
        {
            "label": "Bug",
            "decision": "include",
            "role": "issue_type",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "bug",
        },
        {
            "label": "Docs",
            "decision": "include",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "selected_target",
            "explanation": "docs",
        },
        {
            "label": "Groupby",
            "decision": "defer",
            "role": "component",
            "leakage_risk": "low",
            "reason_code": "insufficient_recent_support",
            "explanation": "groupby",
        },
        {
            "label": "Needs Triage",
            "decision": "exclude",
            "role": "workflow",
            "leakage_risk": "high",
            "reason_code": "workflow_label",
            "explanation": "workflow",
        },
    ]


@dataclass
class Fixture:
    repository: RepositoryRef
    dataset_id: str
    policy_id: str
    split_config_path: Path
    processed_root: Path
    policies_root: Path
    model_ready_root: Path
    processed_manifest: ProcessedManifest


def _setup(tmp_path: Path) -> Fixture:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    policies_root = tmp_path / "policies"
    model_ready_root = tmp_path / "model_ready"
    dataset_dir, dataset_id = write_processed_dataset(processed_root, repository, _issues())
    processed_manifest = ProcessedManifest.model_validate_json(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    config_path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=_labels(),
        selection_criteria=_CRITERIA,
    )
    audit = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    policy = build_label_policy(
        repository,
        dataset_id,
        audit.manifest.audit_id,
        config_path,
        processed_root=processed_root,
        audits_root=audits_root,
        policies_root=policies_root,
    )
    split_config_path = write_temporal_split_config(
        tmp_path / "split.json",
        validation_start="2026-02-01T00:00:00Z",
        test_start="2026-04-01T00:00:00Z",
    )
    return Fixture(
        repository=repository,
        dataset_id=dataset_id,
        policy_id=policy.manifest.policy_id,
        split_config_path=split_config_path,
        processed_root=processed_root,
        policies_root=policies_root,
        model_ready_root=model_ready_root,
        processed_manifest=processed_manifest,
    )


def test_build_publishes_artifact(tmp_path: Path) -> None:
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
    assert not result.cache_hit
    assert result.manifest.records_written == 10
    assert result.manifest.target_count == 2
    assert result.model_dataset_dir.is_dir()
    assert (result.model_dataset_dir / "records.jsonl").is_file()


def test_cache_hit_on_repeated_build(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    first = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    second = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.manifest.model_dataset_id == second.manifest.model_dataset_id


def test_missing_dataset_rejected(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    with pytest.raises(ModelDatasetInputError, match="No normalized dataset"):
        build_model_dataset(
            fixture.repository,
            "20260628T161306010651Z-n1-ffffffffffff",
            fixture.policy_id,
            fixture.split_config_path,
            processed_root=fixture.processed_root,
            policies_root=fixture.policies_root,
            model_ready_root=fixture.model_ready_root,
        )


def test_missing_policy_rejected(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    with pytest.raises(ModelDatasetInputError, match="No label policy"):
        build_model_dataset(
            fixture.repository,
            fixture.dataset_id,
            "20260628T161306010651Z-n1-074402d21505-lp2-ffffffffffff",
            fixture.split_config_path,
            processed_root=fixture.processed_root,
            policies_root=fixture.policies_root,
            model_ready_root=fixture.model_ready_root,
        )


def test_corrupt_cache_rejected(tmp_path: Path) -> None:
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
    records_path.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ModelDatasetCorruptionError):
        build_model_dataset(
            fixture.repository,
            fixture.dataset_id,
            fixture.policy_id,
            fixture.split_config_path,
            processed_root=fixture.processed_root,
            policies_root=fixture.policies_root,
            model_ready_root=fixture.model_ready_root,
        )


def test_immutable_overwrite_refused(tmp_path: Path) -> None:
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
    staging = result.model_dataset_dir.parent / ".staging"
    staging.mkdir()
    with pytest.raises(ModelDatasetBuildError, match="Refusing to overwrite"):
        publish_model_dataset(staging, result.model_dataset_dir)


def test_zero_test_positives_rejected(tmp_path: Path) -> None:
    repository = RepositoryRef(owner="pandas-dev", name="pandas")
    processed_root = tmp_path / "processed"
    audits_root = tmp_path / "audits"
    policies_root = tmp_path / "policies"
    issues = [
        make_normalized_issue(1, labels=["Rare"], created_at=_at(2026, 6, 1)),
        make_normalized_issue(2, labels=["Rare"], created_at=_at(2026, 6, 2)),
    ]
    _, dataset_id = write_processed_dataset(processed_root, repository, issues)
    config_path = write_label_policy_config(
        tmp_path / "policy.json",
        labels=[
            {
                "label": "Rare",
                "decision": "include",
                "role": "component",
                "leakage_risk": "low",
                "reason_code": "selected_target",
                "explanation": "rare",
                "criteria_override_explanation": "test override",
            }
        ],
        selection_criteria={
            "min_total_support": 1,
            "min_active_months": 1,
            "min_recent_support": 1,
            "recent_window_months": 1,
        },
    )
    audit = audit_dataset(
        repository, dataset_id, processed_root=processed_root, audits_root=audits_root
    )
    policy = build_label_policy(
        repository,
        dataset_id,
        audit.manifest.audit_id,
        config_path,
        processed_root=processed_root,
        audits_root=audits_root,
        policies_root=policies_root,
    )
    split_config_path = write_temporal_split_config(
        tmp_path / "split.json",
        validation_start="2026-03-01T00:00:00Z",
        test_start="2026-06-01T00:00:00Z",
    )
    with pytest.raises(ModelDatasetSplitSupportError):
        build_model_dataset(
            repository,
            dataset_id,
            policy.manifest.policy_id,
            split_config_path,
            processed_root=processed_root,
            policies_root=policies_root,
            model_ready_root=tmp_path / "model_ready",
        )


def test_changing_cutoff_changes_id(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    from repotriage.model_dataset.config import load_split_config

    _, config_hash_a = load_split_config(fixture.split_config_path)
    split_b = write_temporal_split_config(
        tmp_path / "split-b.json",
        validation_start="2026-03-01T00:00:00Z",
        test_start="2026-05-01T00:00:00Z",
    )
    _, config_hash_b = load_split_config(split_b)
    policy_dir = fixture.policies_root / fixture.repository.slug / fixture.policy_id
    policy_json_sha256 = hashlib.sha256(
        (policy_dir / "label_policy.json").read_bytes()
    ).hexdigest()
    base = dict(
        model_dataset_version=MODEL_DATASET_VERSION,
        dataset_id=fixture.dataset_id,
        dataset_output_sha256=fixture.processed_manifest.output_sha256,
        policy_id=fixture.policy_id,
        policy_json_sha256=policy_json_sha256,
        text_representation_version=TEXT_REPRESENTATION_VERSION,
        temporal_splitter_version=TEMPORAL_SPLITTER_VERSION,
        split_config_schema_version="1",
    )
    id_a = compute_model_dataset_id(
        fixture.dataset_id,
        compute_model_dataset_input_sha256(**base, split_config_sha256=config_hash_a),
    )
    id_b = compute_model_dataset_id(
        fixture.dataset_id,
        compute_model_dataset_input_sha256(**base, split_config_sha256=config_hash_b),
    )
    assert id_a != id_b


def _validation_context(fixture: Fixture, result) -> dict:
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
    return dict(
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


def _tamper_records_hash_consistent(model_dataset_dir: Path, *, mutate) -> None:
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


def _tamper_label_map_hash_consistent(model_dataset_dir: Path, *, mutate) -> None:
    manifest_path = model_dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_map_path = model_dataset_dir / manifest["label_map_file"]
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    mutate(label_map)
    label_map_bytes = (
        json.dumps(label_map, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    label_map_path.write_bytes(label_map_bytes)
    manifest["label_map_sha256"] = hashlib.sha256(label_map_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_hash_consistent_selected_labels_reorder_rejected(tmp_path: Path) -> None:
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
        if len(record.get("selected_labels", [])) >= 2:
            record["selected_labels"] = list(reversed(record["selected_labels"]))
        else:
            record["target_vector"] = list(reversed(record["target_vector"]))

    _tamper_records_hash_consistent(result.model_dataset_dir, mutate=mutate)
    with pytest.raises(ModelDatasetCorruptionError, match="selected_labels"):
        validate_model_dataset_against_inputs(
            result.model_dataset_dir, **_validation_context(fixture, result)
        )


def test_hash_consistent_label_map_reorder_rejected(tmp_path: Path) -> None:
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

    def mutate(label_map: dict) -> None:
        label_map["labels"] = list(reversed(label_map["labels"]))

    _tamper_label_map_hash_consistent(result.model_dataset_dir, mutate=mutate)
    with pytest.raises(ModelDatasetCorruptionError, match="label_map"):
        validate_model_dataset_against_inputs(
            result.model_dataset_dir, **_validation_context(fixture, result)
        )


def test_hash_consistent_source_divergence_rejected_on_cache_hit(tmp_path: Path) -> None:
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
    from repotriage.model_dataset.text import build_feature_text_v1

    def mutate(record: dict) -> None:
        record["title"] = "Consistent tamper"
        record["feature_text"] = build_feature_text_v1(record["title"], record["body"])

    _tamper_records_hash_consistent(result.model_dataset_dir, mutate=mutate)
    with pytest.raises(ModelDatasetCorruptionError, match="does not derive"):
        build_model_dataset(
            fixture.repository,
            fixture.dataset_id,
            fixture.policy_id,
            fixture.split_config_path,
            processed_root=fixture.processed_root,
            policies_root=fixture.policies_root,
            model_ready_root=fixture.model_ready_root,
        )
