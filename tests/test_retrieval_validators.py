"""Tests for retrieval artifact validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.model_dataset.builder import build_model_dataset
from repotriage.retrieval.builder import build_retrieval_baseline
from repotriage.retrieval.models import RetrievalCorruptionError
from repotriage.retrieval.validators import (
    validate_retrieval_against_model_dataset,
    validate_retrieval_artifact_integrity,
)
from tests.helpers import write_retrieval_baseline_config
from tests.test_model_dataset_builder import _setup


def _build(tmp_path: Path):
    fixture = _setup(tmp_path)
    model_result = build_model_dataset(
        fixture.repository,
        fixture.dataset_id,
        fixture.policy_id,
        fixture.split_config_path,
        processed_root=fixture.processed_root,
        policies_root=fixture.policies_root,
        model_ready_root=fixture.model_ready_root,
    )
    config_path = write_retrieval_baseline_config(tmp_path / "retrieval.json", min_df=1)
    result = build_retrieval_baseline(
        fixture.repository,
        model_result.manifest.model_dataset_id,
        config_path,
        model_ready_root=fixture.model_ready_root,
        retrieval_baselines_root=tmp_path / "retrieval_baselines",
    )
    return fixture, model_result, config_path, result


def test_against_inputs_passes(tmp_path: Path) -> None:
    fixture, model_result, config_path, result = _build(tmp_path)
    manifest = validate_retrieval_against_model_dataset(
        result.retrieval_dir,
        model_result.model_dataset_dir,
        expected_repository=fixture.repository,
        expected_model_dataset_id=model_result.manifest.model_dataset_id,
        config_path=config_path,
    )
    assert manifest.retrieval_run_id == result.manifest.retrieval_run_id


def test_integrity_rejects_tampered_neighbors(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    neighbors_path = result.retrieval_dir / "neighbors_validation.jsonl"
    lines = neighbors_path.read_text(encoding="utf-8").splitlines()
    payload = __import__("json").loads(lines[0])
    payload["neighbors"][0]["similarity"] = 999.0
    lines[0] = __import__("json").dumps(payload)
    neighbors_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RetrievalCorruptionError):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_retrieval_run_id=result.manifest.retrieval_run_id,
        )


def _load_json_lines(path: Path) -> list[dict]:
    import json

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_lines(path: Path, rows: list[dict]) -> None:
    import json

    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _rewrite_manifest_hash_for(path: Path, artifact_dir: Path, field: str) -> None:
    import hashlib
    import json

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_integrity_rejects_duplicate_neighbor_rank(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    rows[0]["neighbors"][1]["rank"] = rows[0]["neighbors"][0]["rank"]
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="not contiguous"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_missing_neighbor_rank(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    rows[0]["neighbors"][1]["rank"] = rows[0]["neighbors"][0]["rank"] + 2
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="not contiguous"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_similarity_out_of_order(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    rows[0]["neighbors"][0]["similarity"] = 0.1
    rows[0]["neighbors"][1]["similarity"] = 0.9
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="descending"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_nan_or_inf_similarity(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    rows[0]["neighbors"][0]["similarity"] = float("inf")
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="Invalid JSONL|not finite"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_neighbor_from_validation_or_test_split(tmp_path: Path) -> None:
    fixture, model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    validation_issue_id = rows[0]["query_issue_id"]
    rows[0]["neighbors"][0]["neighbor_issue_id"] = validation_issue_id
    rows[0]["neighbors"][0]["neighbor_issue_number"] = validation_issue_id
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="not in train corpus|also a query"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )
    # Also prove a test-split id is rejected.
    test_path = result.retrieval_dir / "neighbors_test.jsonl"
    test_rows = _load_json_lines(test_path)
    test_issue_id = test_rows[0]["query_issue_id"]
    val_rows = _load_json_lines(path)
    val_rows[0]["neighbors"][0]["neighbor_issue_id"] = test_issue_id
    val_rows[0]["neighbors"][0]["neighbor_issue_number"] = test_issue_id
    _write_json_lines(path, val_rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="not in train corpus|also a query"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )
    del model_result


def test_integrity_rejects_wrong_neighbor_row_count(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    _write_json_lines(path, rows[:-1])
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="neighbor rows"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_wrong_neighbors_per_query(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "neighbors_validation.jsonl"
    rows = _load_json_lines(path)
    rows[0]["neighbors"] = rows[0]["neighbors"][:-1]
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "neighbors_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="neighbors; expected"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_tampered_metrics_validation(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    import json

    path = result.retrieval_dir / "metrics_validation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["all_zero_label_query_count"] = payload["all_zero_label_query_count"] + 1
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "metrics_validation_sha256")
    with pytest.raises(RetrievalCorruptionError, match="all_zero_label_query_count"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_tampered_corpus_records_order(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "corpus_records.jsonl"
    rows = _load_json_lines(path)
    rows[0], rows[1] = rows[1], rows[0]
    _write_json_lines(path, rows)
    _rewrite_manifest_hash_for(path, result.retrieval_dir, "corpus_records_sha256")
    with pytest.raises(RetrievalCorruptionError, match="issue_id order"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_corrupted_vectorizer_hash(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "vectorizer.joblib"
    path.write_bytes(path.read_bytes() + b"\ncorrupt")
    with pytest.raises(RetrievalCorruptionError, match="Hash mismatch"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )


def test_integrity_rejects_corrupted_corpus_matrix_hash(tmp_path: Path) -> None:
    fixture, _model_result, _config_path, result = _build(tmp_path)
    path = result.retrieval_dir / "corpus_matrix.npz"
    data = bytearray(path.read_bytes())
    data[-1] = (data[-1] + 1) % 256
    path.write_bytes(bytes(data))
    with pytest.raises(RetrievalCorruptionError, match="Hash mismatch"):
        validate_retrieval_artifact_integrity(
            result.retrieval_dir,
            expected_repository=fixture.repository,
            expected_run_id=result.manifest.retrieval_run_id,
        )
