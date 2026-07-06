"""Integrity and against-inputs validation for retrieval-baseline artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import ValidationError

from repotriage.baseline.reader import load_test_split, load_training_splits
from repotriage.github.models import RepositoryRef
from repotriage.model_dataset.builder import validate_model_dataset_artifact_integrity
from repotriage.paths import resolve_within_directory
from repotriage.retrieval.evaluator import compute_retrieval_metrics
from repotriage.retrieval.index import (
    build_retrieval_index,
    compute_index_semantic_sha256,
    load_corpus_matrix,
)
from repotriage.retrieval.models import (
    CORPUS_MATRIX_NPZ_FILE,
    CORPUS_RECORDS_JSONL_FILE,
    MANIFEST_JSON_FILE,
    VECTORIZER_JOBLIB_FILE,
    CorpusRecord,
    FrozenRetrievalConfig,
    IndexMetadataDocument,
    QueryNeighborRecord,
    RetrievalCorruptionError,
    RetrievalInputError,
    RetrievalManifest,
    RetrievalMetricsDocument,
    compute_retrieval_experiment_sha256,
    compute_retrieval_run_id,
    compute_retrieval_run_sha256,
    floats_consistent,
)
from repotriage.retrieval.report import sha256_hex
from repotriage.retrieval.search import assert_neighbors_descending, search_split


def _load_manifest(retrieval_dir: Path) -> RetrievalManifest:
    manifest_path = resolve_within_directory(retrieval_dir, MANIFEST_JSON_FILE)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalCorruptionError(
            f"Unable to read retrieval manifest at {manifest_path}: {exc}"
        ) from exc
    try:
        return RetrievalManifest.model_validate(payload)
    except ValidationError as exc:
        raise RetrievalCorruptionError(
            f"Invalid retrieval manifest at {manifest_path}: {exc}"
        ) from exc


def _verify_file_hash(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise RetrievalCorruptionError(f"Missing artifact file: {path}")
    data = path.read_bytes()
    actual = sha256_hex(data)
    if actual != expected_sha256:
        raise RetrievalCorruptionError(
            f"Hash mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return data


def _read_jsonl(path: Path, model: type) -> list:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise RetrievalCorruptionError(
                f"Invalid JSONL record at {path}:{line_number}: {exc}"
            ) from exc
    return records


def _validate_neighbors_file(
    path: Path,
    *,
    expected_count: int,
    top_k: int,
    corpus_size: int,
    train_issue_ids: set[int],
    query_issue_ids: set[int],
) -> list[QueryNeighborRecord]:
    expected_neighbors = min(top_k, corpus_size)
    records = _read_jsonl(path, QueryNeighborRecord)
    if len(records) != expected_count:
        raise RetrievalCorruptionError(
            f"Expected {expected_count} neighbor rows in {path.name}; got {len(records)}"
        )
    seen_queries: set[int] = set()
    for record in records:
        if record.query_issue_id in seen_queries:
            raise RetrievalCorruptionError(
                f"Duplicate query_issue_id in {path.name}: {record.query_issue_id}"
            )
        seen_queries.add(record.query_issue_id)
        if len(record.neighbors) != expected_neighbors:
            raise RetrievalCorruptionError(
                f"Query {record.query_issue_id} has {len(record.neighbors)} neighbors; "
                f"expected {expected_neighbors}"
            )
        assert_neighbors_descending(record.neighbors)
        neighbor_ids: set[int] = set()
        for neighbor in record.neighbors:
            if not math.isfinite(neighbor.similarity):
                raise RetrievalCorruptionError("Neighbor similarity is not finite")
            if neighbor.neighbor_issue_id in neighbor_ids:
                raise RetrievalCorruptionError(
                    f"Duplicate neighbor for query {record.query_issue_id}: "
                    f"{neighbor.neighbor_issue_id}"
                )
            neighbor_ids.add(neighbor.neighbor_issue_id)
            if neighbor.neighbor_issue_id not in train_issue_ids:
                raise RetrievalCorruptionError(
                    f"Neighbor issue_id {neighbor.neighbor_issue_id} is not in train corpus"
                )
            if neighbor.neighbor_issue_id in query_issue_ids:
                raise RetrievalCorruptionError(
                    f"Neighbor issue_id {neighbor.neighbor_issue_id} is also a query"
                )
    return records


def _metrics_close(left: RetrievalMetricsDocument, right: RetrievalMetricsDocument) -> None:
    for field in (
        "total_query_count",
        "scored_query_count",
        "all_zero_label_query_count",
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "precision_at_10",
        "mrr_at_10",
        "mean_best_label_jaccard_at_10",
        "mean_best_shared_label_count_at_10",
    ):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if isinstance(left_value, int):
            if left_value != right_value:
                raise RetrievalCorruptionError(
                    f"Metric {field} mismatch: {left_value} vs {right_value}"
                )
        elif not floats_consistent(left_value, right_value):
            raise RetrievalCorruptionError(
                f"Metric {field} mismatch: {left_value} vs {right_value}"
            )


def _neighbors_close(
    stored: list[QueryNeighborRecord],
    recomputed: list[QueryNeighborRecord],
) -> None:
    if len(stored) != len(recomputed):
        raise RetrievalCorruptionError("Neighbor row count mismatch on recompute")
    for stored_row, recomputed_row in zip(stored, recomputed, strict=True):
        if stored_row.query_issue_id != recomputed_row.query_issue_id:
            raise RetrievalCorruptionError("Query issue_id mismatch on recompute")
        if len(stored_row.neighbors) != len(recomputed_row.neighbors):
            raise RetrievalCorruptionError("Neighbor count mismatch on recompute")
        for stored_neighbor, recomputed_neighbor in zip(
            stored_row.neighbors,
            recomputed_row.neighbors,
            strict=True,
        ):
            if stored_neighbor.neighbor_issue_id != recomputed_neighbor.neighbor_issue_id:
                raise RetrievalCorruptionError("Neighbor issue_id mismatch on recompute")
            if stored_neighbor.rank != recomputed_neighbor.rank:
                raise RetrievalCorruptionError("Neighbor rank mismatch on recompute")
            if not floats_consistent(
                stored_neighbor.similarity,
                recomputed_neighbor.similarity,
            ):
                raise RetrievalCorruptionError(
                    f"Similarity mismatch for query {stored_row.query_issue_id} "
                    f"neighbor {stored_neighbor.neighbor_issue_id}"
                )


def validate_retrieval_artifact_integrity(
    retrieval_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_retrieval_run_id: str | None = None,
    expected_run_id: str | None = None,
    check_dir_name: bool = True,
) -> RetrievalManifest:
    """Validate a retrieval artifact using on-disk bytes and structural checks."""
    if not retrieval_dir.is_dir():
        raise RetrievalCorruptionError(f"Retrieval directory does not exist: {retrieval_dir}")

    manifest = _load_manifest(retrieval_dir)
    if check_dir_name and retrieval_dir.name != manifest.retrieval_run_id:
        raise RetrievalCorruptionError(
            f"Directory name {retrieval_dir.name!r} does not match manifest "
            f"retrieval_run_id {manifest.retrieval_run_id!r}."
        )
    resolved_expected_run_id = expected_retrieval_run_id or expected_run_id
    if resolved_expected_run_id is None:
        raise RetrievalCorruptionError(
            "Expected retrieval run id is required (expected_retrieval_run_id or expected_run_id)."
        )
    if manifest.retrieval_run_id != resolved_expected_run_id:
        raise RetrievalCorruptionError(
            f"Manifest retrieval_run_id {manifest.retrieval_run_id!r} does not match expected "
            f"{resolved_expected_run_id!r}."
        )
    if manifest.repository != expected_repository.full_name:
        raise RetrievalCorruptionError(
            f"Manifest repository {manifest.repository!r} does not match expected "
            f"{expected_repository.full_name!r}."
        )

    expected_run_sha256 = compute_retrieval_run_sha256(
        manifest.retrieval_experiment_sha256,
        manifest.numerical_environment_sha256,
    )
    if manifest.retrieval_run_sha256 != expected_run_sha256:
        raise RetrievalCorruptionError("retrieval_run_sha256 is inconsistent with manifest fields")

    expected_run_id = compute_retrieval_run_id(
        manifest.model_dataset_id,
        expected_run_sha256,
        manifest.retrieval_baseline_version,
    )
    if manifest.retrieval_run_id != expected_run_id:
        raise RetrievalCorruptionError(
            f"retrieval_run_id {manifest.retrieval_run_id!r} is inconsistent with inputs "
            f"(expected {expected_run_id!r})."
        )

    for relative_path, expected_sha256 in (
        (manifest.config_file, manifest.config_sha256),
        (manifest.index_metadata_file, manifest.index_metadata_sha256),
        (manifest.corpus_records_file, manifest.corpus_records_sha256),
        (manifest.neighbors_validation_file, manifest.neighbors_validation_sha256),
        (manifest.neighbors_test_file, manifest.neighbors_test_sha256),
        (manifest.metrics_validation_file, manifest.metrics_validation_sha256),
        (manifest.metrics_test_file, manifest.metrics_test_sha256),
        (manifest.report_file, manifest.report_sha256),
        (manifest.vectorizer_file, manifest.vectorizer_sha256),
        (manifest.corpus_matrix_file, manifest.corpus_matrix_sha256),
    ):
        _verify_file_hash(resolve_within_directory(retrieval_dir, relative_path), expected_sha256)

    index_metadata = IndexMetadataDocument.model_validate_json(
        _verify_file_hash(
            resolve_within_directory(retrieval_dir, manifest.index_metadata_file),
            manifest.index_metadata_sha256,
        )
    )
    if index_metadata.index_semantic_sha256 != manifest.index_semantic_sha256:
        raise RetrievalCorruptionError(
            "index_metadata index_semantic_sha256 does not match manifest"
        )
    if index_metadata.corpus_size != manifest.corpus_size:
        raise RetrievalCorruptionError("index_metadata corpus_size does not match manifest")
    if list(index_metadata.train_issue_ids) != list(manifest.train_issue_ids):
        raise RetrievalCorruptionError("index_metadata train_issue_ids does not match manifest")

    corpus_records = _read_jsonl(
        resolve_within_directory(retrieval_dir, manifest.corpus_records_file),
        CorpusRecord,
    )
    if len(corpus_records) != manifest.corpus_size:
        raise RetrievalCorruptionError(
            f"corpus_records count {len(corpus_records)} does not match manifest "
            f"{manifest.corpus_size}"
        )
    expected_corpus_ids = list(manifest.train_issue_ids)
    actual_corpus_ids = [record.issue_id for record in corpus_records]
    if actual_corpus_ids != expected_corpus_ids:
        raise RetrievalCorruptionError(
            "corpus_records issue_id order does not match manifest train_issue_ids"
        )
    expected_indices = list(range(len(corpus_records)))
    actual_indices = [record.corpus_index for record in corpus_records]
    if actual_indices != expected_indices:
        raise RetrievalCorruptionError(
            "corpus_records corpus_index values must be contiguous from 0"
        )

    train_issue_ids = set(manifest.train_issue_ids)
    val_path = resolve_within_directory(retrieval_dir, manifest.neighbors_validation_file)
    test_path = resolve_within_directory(retrieval_dir, manifest.neighbors_test_file)
    val_records = _read_jsonl(val_path, QueryNeighborRecord)
    test_records = _read_jsonl(test_path, QueryNeighborRecord)
    _validate_neighbors_file(
        val_path,
        expected_count=manifest.validation_query_count,
        top_k=manifest.top_k,
        corpus_size=manifest.corpus_size,
        train_issue_ids=train_issue_ids,
        query_issue_ids={record.query_issue_id for record in val_records},
    )
    _validate_neighbors_file(
        test_path,
        expected_count=manifest.test_query_count,
        top_k=manifest.top_k,
        corpus_size=manifest.corpus_size,
        train_issue_ids=train_issue_ids,
        query_issue_ids={record.query_issue_id for record in test_records},
    )

    all_zero_val = sum(1 for record in val_records if not record.query_has_positive_labels)
    metrics_validation = RetrievalMetricsDocument.model_validate_json(
        resolve_within_directory(retrieval_dir, manifest.metrics_validation_file).read_text(
            encoding="utf-8"
        )
    )
    if metrics_validation.all_zero_label_query_count != all_zero_val:
        raise RetrievalCorruptionError(
            "metrics_validation all_zero_label_query_count does not match neighbors file"
        )

    return manifest


def validate_retrieval_against_model_dataset(
    retrieval_dir: Path,
    model_dataset_dir: Path,
    *,
    expected_repository: RepositoryRef,
    expected_model_dataset_id: str,
    config_path: Path | None = None,
    expected_retrieval_run_id: str | None = None,
    expected_retrieval_experiment_sha256: str | None = None,
    expected_numerical_environment_sha256: str | None = None,
) -> RetrievalManifest:
    """Recompute retrieval outputs from model-ready inputs and compare to the artifact."""
    resolved_expected_run_id = expected_retrieval_run_id or retrieval_dir.name
    manifest = validate_retrieval_artifact_integrity(
        retrieval_dir,
        expected_repository=expected_repository,
        expected_retrieval_run_id=resolved_expected_run_id,
        check_dir_name=retrieval_dir.name == resolved_expected_run_id,
    )
    if manifest.model_dataset_id != expected_model_dataset_id:
        raise RetrievalCorruptionError(
            f"Manifest model_dataset_id {manifest.model_dataset_id!r} does not match expected "
            f"{expected_model_dataset_id!r}."
        )

    _md_manifest, label_map, _split_report = validate_model_dataset_artifact_integrity(
        model_dataset_dir,
        expected_repository=expected_repository,
        expected_model_dataset_id=expected_model_dataset_id,
    )
    if _md_manifest.records_sha256 != manifest.records_sha256:
        raise RetrievalCorruptionError("records_sha256 does not match model-ready manifest")
    if _md_manifest.label_map_sha256 != manifest.label_map_sha256:
        raise RetrievalCorruptionError("label_map_sha256 does not match model-ready manifest")

    frozen_config = FrozenRetrievalConfig.model_validate_json(
        resolve_within_directory(retrieval_dir, manifest.config_file).read_text(encoding="utf-8")
    )
    if config_path is not None:
        from repotriage.retrieval.config import load_retrieval_config

        config, _, _, semantic_hash = load_retrieval_config(config_path)
        if semantic_hash != manifest.config_semantic_sha256:
            raise RetrievalCorruptionError("config_semantic_sha256 does not match provided config")
        if config.repository != manifest.repository:
            raise RetrievalCorruptionError("config repository does not match manifest")

    expected_experiment = compute_retrieval_experiment_sha256(
        retrieval_baseline_version=manifest.retrieval_baseline_version,
        retrieval_protocol_version=manifest.retrieval_protocol_version,
        metric_contract_version=manifest.metric_contract_version,
        model_dataset_id=manifest.model_dataset_id,
        records_sha256=manifest.records_sha256,
        label_map_sha256=manifest.label_map_sha256,
        config_semantic_sha256=manifest.config_semantic_sha256,
        top_k=manifest.top_k,
        similarity_metric=manifest.similarity_metric,
        relevance_definition=manifest.relevance_definition,
        tfidf=frozen_config.tfidf,
        label_order=label_map.labels,
    )
    if manifest.retrieval_experiment_sha256 != expected_experiment:
        raise RetrievalCorruptionError(
            "retrieval_experiment_sha256 does not match recomputed value"
        )
    if (
        expected_retrieval_experiment_sha256 is not None
        and manifest.retrieval_experiment_sha256 != expected_retrieval_experiment_sha256
    ):
        raise RetrievalCorruptionError(
            "retrieval_experiment_sha256 does not match expected pre-publish value"
        )
    if (
        expected_numerical_environment_sha256 is not None
        and manifest.numerical_environment_sha256 != expected_numerical_environment_sha256
    ):
        raise RetrievalCorruptionError(
            "numerical_environment_sha256 does not match expected pre-publish value"
        )

    training_splits = load_training_splits(
        model_dataset_dir,
        expected_repository=expected_repository,
        expected_model_dataset_id=expected_model_dataset_id,
    )
    test_split = load_test_split(
        model_dataset_dir,
        manifest=_md_manifest,
        label_map=label_map,
        training_splits=training_splits,
    )

    index = build_retrieval_index(training_splits.train, frozen_config.tfidf)
    recomputed_index_hash = compute_index_semantic_sha256(
        index,
        retrieval_baseline_version=manifest.retrieval_baseline_version,
        model_dataset_id=manifest.model_dataset_id,
        repository=manifest.repository,
        label_order=label_map.labels,
        top_k=manifest.top_k,
        similarity_metric=manifest.similarity_metric,
        metric_contract_version=manifest.metric_contract_version,
        numerical_environment_sha256=manifest.numerical_environment_sha256,
    )
    if recomputed_index_hash != manifest.index_semantic_sha256:
        raise RetrievalCorruptionError("index_semantic_sha256 does not match recomputed index")

    train_ids = set(index.train_issue_ids)
    val_ids = {record.issue_id for record in training_splits.validation.records}
    test_ids = {record.issue_id for record in test_split.records}
    if train_ids & val_ids or train_ids & test_ids:
        raise RetrievalInputError("Train issue ids overlap validation or test splits")

    val_neighbors = search_split(
        index,
        training_splits.validation,
        split_name="validation",
        top_k=manifest.top_k,
        label_order=label_map.labels,
    )
    test_neighbors = search_split(
        index,
        test_split,
        split_name="test",
        top_k=manifest.top_k,
        label_order=label_map.labels,
    )
    val_metrics = compute_retrieval_metrics(val_neighbors, split="validation")
    test_metrics = compute_retrieval_metrics(test_neighbors, split="test")

    stored_val_neighbors = _read_jsonl(
        resolve_within_directory(retrieval_dir, manifest.neighbors_validation_file),
        QueryNeighborRecord,
    )
    stored_test_neighbors = _read_jsonl(
        resolve_within_directory(retrieval_dir, manifest.neighbors_test_file),
        QueryNeighborRecord,
    )
    stored_val_metrics = RetrievalMetricsDocument.model_validate_json(
        resolve_within_directory(retrieval_dir, manifest.metrics_validation_file).read_text(
            encoding="utf-8"
        )
    )
    stored_test_metrics = RetrievalMetricsDocument.model_validate_json(
        resolve_within_directory(retrieval_dir, manifest.metrics_test_file).read_text(
            encoding="utf-8"
        )
    )

    _neighbors_close(stored_val_neighbors, val_neighbors)
    _neighbors_close(stored_test_neighbors, test_neighbors)
    _metrics_close(stored_val_metrics, val_metrics)
    _metrics_close(stored_test_metrics, test_metrics)

    return manifest


def verify_retrieval_index_consistency(
    retrieval_dir: Path,
    manifest: RetrievalManifest,
    *,
    label_order: list[str],
) -> None:
    """Load serialized index files and verify they match the semantic fingerprint."""
    import joblib

    from repotriage.retrieval.index import RetrievalIndex

    vectorizer = joblib.load(resolve_within_directory(retrieval_dir, VECTORIZER_JOBLIB_FILE))
    corpus_matrix = load_corpus_matrix(
        str(resolve_within_directory(retrieval_dir, CORPUS_MATRIX_NPZ_FILE))
    )
    corpus_records = _read_jsonl(
        resolve_within_directory(retrieval_dir, CORPUS_RECORDS_JSONL_FILE),
        CorpusRecord,
    )
    loaded_index = RetrievalIndex(
        vectorizer=vectorizer,
        corpus_matrix=corpus_matrix,
        corpus_records=corpus_records,
        train_issue_ids=[record.issue_id for record in corpus_records],
    )
    recomputed = compute_index_semantic_sha256(
        loaded_index,
        retrieval_baseline_version=manifest.retrieval_baseline_version,
        model_dataset_id=manifest.model_dataset_id,
        repository=manifest.repository,
        label_order=label_order,
        top_k=manifest.top_k,
        similarity_metric=manifest.similarity_metric,
        metric_contract_version=manifest.metric_contract_version,
        numerical_environment_sha256=manifest.numerical_environment_sha256,
    )
    if recomputed != manifest.index_semantic_sha256:
        raise RetrievalCorruptionError(
            "Loaded vectorizer/matrix semantic fingerprint does not match manifest"
        )
