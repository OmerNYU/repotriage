"""Build (or reuse) an immutable retrieval-baseline artifact."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from repotriage.baseline.builder import collect_environment_metadata_and_hash
from repotriage.baseline.reader import load_test_split, load_training_splits
from repotriage.filesystem import atomic_write_bytes, best_effort_remove_tree
from repotriage.github.models import RepositoryRef
from repotriage.retrieval.config import load_retrieval_config
from repotriage.retrieval.evaluator import compute_retrieval_metrics
from repotriage.retrieval.index import (
    build_retrieval_index,
    compute_index_semantic_sha256,
    save_corpus_matrix,
)
from repotriage.retrieval.models import (
    CONFIG_JSON_FILE,
    CORPUS_MATRIX_NPZ_FILE,
    CORPUS_RECORDS_JSONL_FILE,
    INDEX_METADATA_JSON_FILE,
    MANIFEST_JSON_FILE,
    METRICS_TEST_JSON_FILE,
    METRICS_VALIDATION_JSON_FILE,
    NEIGHBORS_TEST_JSONL_FILE,
    NEIGHBORS_VALIDATION_JSONL_FILE,
    REPORT_MARKDOWN_FILE,
    VECTORIZER_JOBLIB_FILE,
    FrozenRetrievalConfig,
    IndexMetadataDocument,
    RetrievalBuildError,
    RetrievalInputError,
    RetrievalManifest,
    RetrievalMetricsDocument,
    compute_retrieval_experiment_sha256,
    compute_retrieval_run_id,
    compute_retrieval_run_sha256,
)
from repotriage.retrieval.report import (
    serialize_config_json,
    serialize_corpus_records_jsonl,
    serialize_index_metadata_json,
    serialize_manifest_json,
    serialize_metrics_json,
    serialize_query_neighbors_jsonl,
    serialize_report_markdown,
    sha256_hex,
)
from repotriage.retrieval.search import search_split
from repotriage.retrieval.validators import (
    validate_retrieval_against_model_dataset,
    validate_retrieval_artifact_integrity,
    verify_retrieval_index_consistency,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_READY_ROOT = Path("data/model_ready/github")
DEFAULT_RETRIEVAL_BASELINES_ROOT = Path("data/retrieval_baselines/github")


@dataclass(frozen=True)
class RetrievalBuildResult:
    repository: RepositoryRef
    retrieval_dir: Path
    manifest: RetrievalManifest
    validation_metrics: RetrievalMetricsDocument
    test_metrics: RetrievalMetricsDocument
    cache_hit: bool


def publish_retrieval_baseline(staging_dir: Path, final_dir: Path) -> None:
    """Atomically publish a staged retrieval artifact directory."""
    if final_dir.exists():
        raise RetrievalBuildError(f"Refusing to overwrite existing retrieval artifact: {final_dir}")
    try:
        os.rename(staging_dir, final_dir)
    except OSError as exc:
        raise RetrievalBuildError(
            f"Unable to publish retrieval artifact from {staging_dir} to {final_dir}: {exc}"
        ) from exc


def build_retrieval_baseline(
    repository: RepositoryRef,
    model_dataset_id: str,
    config_path: Path,
    *,
    model_ready_root: Path = DEFAULT_MODEL_READY_ROOT,
    retrieval_baselines_root: Path = DEFAULT_RETRIEVAL_BASELINES_ROOT,
) -> RetrievalBuildResult:
    """Build or reuse one immutable retrieval-baseline artifact."""
    config, _config_bytes, config_source_hash, config_semantic_hash = load_retrieval_config(
        config_path
    )
    if config.repository != repository.full_name:
        raise RetrievalInputError(
            f"Config repository {config.repository!r} does not match requested "
            f"repository {repository.full_name!r}."
        )

    model_dataset_dir = model_ready_root / repository.slug / model_dataset_id
    if not model_dataset_dir.is_dir():
        raise RetrievalInputError(
            f"No model-ready artifact found for {repository.full_name} with model dataset id "
            f"{model_dataset_id!r} at {model_dataset_dir}."
        )

    training_splits = load_training_splits(
        model_dataset_dir,
        expected_repository=repository,
        expected_model_dataset_id=model_dataset_id,
    )
    manifest_md = training_splits.manifest
    label_map = training_splits.label_map
    labels = label_map.labels

    _environment, env_hash = collect_environment_metadata_and_hash()

    retrieval_experiment_sha256 = compute_retrieval_experiment_sha256(
        retrieval_baseline_version=config.retrieval_baseline_version,
        retrieval_protocol_version=config.retrieval_protocol_version,
        metric_contract_version=config.metric_contract_version,
        model_dataset_id=model_dataset_id,
        records_sha256=manifest_md.records_sha256,
        label_map_sha256=manifest_md.label_map_sha256,
        config_semantic_sha256=config_semantic_hash,
        top_k=config.top_k,
        similarity_metric=config.similarity_metric,
        relevance_definition=config.relevance_definition,
        tfidf=config.tfidf,
        label_order=labels,
    )
    retrieval_run_sha256 = compute_retrieval_run_sha256(
        retrieval_experiment_sha256,
        env_hash,
    )
    retrieval_run_id = compute_retrieval_run_id(model_dataset_id, retrieval_run_sha256)
    final_dir = retrieval_baselines_root / repository.slug / retrieval_run_id

    if final_dir.exists():
        published = validate_retrieval_against_model_dataset(
            final_dir,
            model_dataset_dir,
            expected_repository=repository,
            expected_model_dataset_id=model_dataset_id,
            config_path=config_path,
            expected_retrieval_run_id=retrieval_run_id,
            expected_retrieval_experiment_sha256=retrieval_experiment_sha256,
            expected_numerical_environment_sha256=env_hash,
        )
        val_metrics = RetrievalMetricsDocument.model_validate_json(
            (final_dir / METRICS_VALIDATION_JSON_FILE).read_text(encoding="utf-8")
        )
        test_metrics = RetrievalMetricsDocument.model_validate_json(
            (final_dir / METRICS_TEST_JSON_FILE).read_text(encoding="utf-8")
        )
        logger.info("Retrieval-baseline cache hit for %s at %s", repository.full_name, final_dir)
        return RetrievalBuildResult(
            repository=repository,
            retrieval_dir=final_dir,
            manifest=published,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            cache_hit=True,
        )

    index = build_retrieval_index(training_splits.train, config.tfidf)
    index_semantic_sha256 = compute_index_semantic_sha256(
        index,
        retrieval_baseline_version=config.retrieval_baseline_version,
        model_dataset_id=model_dataset_id,
        repository=repository.full_name,
        label_order=labels,
        top_k=config.top_k,
        similarity_metric=config.similarity_metric,
        metric_contract_version=config.metric_contract_version,
        numerical_environment_sha256=env_hash,
    )

    val_neighbors = search_split(
        index,
        training_splits.validation,
        split_name="validation",
        top_k=config.top_k,
        label_order=labels,
    )
    val_metrics = compute_retrieval_metrics(val_neighbors, split="validation")

    test_split = load_test_split(
        model_dataset_dir,
        manifest=manifest_md,
        label_map=label_map,
        training_splits=training_splits,
    )
    test_neighbors = search_split(
        index,
        test_split,
        split_name="test",
        top_k=config.top_k,
        label_order=labels,
    )
    test_metrics = compute_retrieval_metrics(test_neighbors, split="test")

    frozen_config = FrozenRetrievalConfig(
        repository=config.repository,
        retrieval_protocol_version=config.retrieval_protocol_version,
        metric_contract_version=config.metric_contract_version,
        similarity_metric=config.similarity_metric,
        relevance_definition=config.relevance_definition,
        top_k=config.top_k,
        tfidf=config.tfidf,
    )
    index_metadata = IndexMetadataDocument(
        index_semantic_sha256=index_semantic_sha256,
        model_dataset_id=model_dataset_id,
        repository=repository.full_name,
        corpus_size=index.corpus_size,
        vocabulary_size=index.vocabulary_size,
        matrix_shape=(index.corpus_matrix.shape[0], index.corpus_matrix.shape[1]),
        train_issue_ids=list(index.train_issue_ids),
        top_k=config.top_k,
        similarity_metric=config.similarity_metric,
    )

    config_json = serialize_config_json(frozen_config)
    index_metadata_json = serialize_index_metadata_json(index_metadata)
    corpus_records_jsonl = serialize_corpus_records_jsonl(index.corpus_records)
    neighbors_validation_jsonl = serialize_query_neighbors_jsonl(val_neighbors)
    neighbors_test_jsonl = serialize_query_neighbors_jsonl(test_neighbors)
    metrics_validation_json = serialize_metrics_json(val_metrics)
    metrics_test_json = serialize_metrics_json(test_metrics)
    report_md = serialize_report_markdown(
        repository=repository.full_name,
        model_dataset_id=model_dataset_id,
        retrieval_run_id=retrieval_run_id,
        corpus_size=index.corpus_size,
        validation_query_count=len(val_neighbors),
        test_query_count=len(test_neighbors),
        top_k=config.top_k,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        index_semantic_sha256=index_semantic_sha256,
    )

    vectorizer_buffer = io.BytesIO()
    joblib.dump(index.vectorizer, vectorizer_buffer)
    vectorizer_bytes = vectorizer_buffer.getvalue()

    matrix_buffer = io.BytesIO()
    save_corpus_matrix(matrix_buffer, index.corpus_matrix)
    matrix_bytes = matrix_buffer.getvalue()

    built_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    manifest = RetrievalManifest(
        retrieval_run_id=retrieval_run_id,
        retrieval_experiment_sha256=retrieval_experiment_sha256,
        numerical_environment_sha256=env_hash,
        retrieval_run_sha256=retrieval_run_sha256,
        config_source_sha256=config_source_hash,
        config_semantic_sha256=config_semantic_hash,
        repository=repository.full_name,
        model_dataset_id=model_dataset_id,
        records_sha256=manifest_md.records_sha256,
        label_map_sha256=manifest_md.label_map_sha256,
        retrieval_protocol_version=config.retrieval_protocol_version,
        metric_contract_version=config.metric_contract_version,
        similarity_metric=config.similarity_metric,
        relevance_definition=config.relevance_definition,
        top_k=config.top_k,
        corpus_size=index.corpus_size,
        validation_query_count=len(val_neighbors),
        test_query_count=len(test_neighbors),
        target_count=label_map.target_count,
        train_issue_ids=list(index.train_issue_ids),
        index_semantic_sha256=index_semantic_sha256,
        built_at=built_at,
        config_sha256=sha256_hex(config_json),
        index_metadata_sha256=sha256_hex(index_metadata_json),
        corpus_records_sha256=sha256_hex(corpus_records_jsonl),
        neighbors_validation_sha256=sha256_hex(neighbors_validation_jsonl),
        neighbors_test_sha256=sha256_hex(neighbors_test_jsonl),
        metrics_validation_sha256=sha256_hex(metrics_validation_json),
        metrics_test_sha256=sha256_hex(metrics_test_json),
        report_sha256=sha256_hex(report_md),
        vectorizer_sha256=sha256_hex(vectorizer_bytes),
        corpus_matrix_sha256=sha256_hex(matrix_bytes),
    )
    manifest_json = serialize_manifest_json(manifest)

    slug_dir = retrieval_baselines_root / repository.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(slug_dir)))

    try:
        atomic_write_bytes(staging_dir / CONFIG_JSON_FILE, config_json)
        atomic_write_bytes(staging_dir / INDEX_METADATA_JSON_FILE, index_metadata_json)
        atomic_write_bytes(staging_dir / CORPUS_RECORDS_JSONL_FILE, corpus_records_jsonl)
        atomic_write_bytes(
            staging_dir / NEIGHBORS_VALIDATION_JSONL_FILE,
            neighbors_validation_jsonl,
        )
        atomic_write_bytes(staging_dir / NEIGHBORS_TEST_JSONL_FILE, neighbors_test_jsonl)
        atomic_write_bytes(staging_dir / METRICS_VALIDATION_JSON_FILE, metrics_validation_json)
        atomic_write_bytes(staging_dir / METRICS_TEST_JSON_FILE, metrics_test_json)
        atomic_write_bytes(staging_dir / REPORT_MARKDOWN_FILE, report_md)
        atomic_write_bytes(staging_dir / VECTORIZER_JOBLIB_FILE, vectorizer_bytes)
        atomic_write_bytes(staging_dir / CORPUS_MATRIX_NPZ_FILE, matrix_bytes)
        atomic_write_bytes(staging_dir / MANIFEST_JSON_FILE, manifest_json)

        validate_retrieval_artifact_integrity(
            staging_dir,
            expected_repository=repository,
            expected_retrieval_run_id=retrieval_run_id,
            check_dir_name=False,
        )
        validate_retrieval_against_model_dataset(
            staging_dir,
            model_dataset_dir,
            expected_repository=repository,
            expected_model_dataset_id=model_dataset_id,
            config_path=config_path,
            expected_retrieval_run_id=retrieval_run_id,
            expected_retrieval_experiment_sha256=retrieval_experiment_sha256,
            expected_numerical_environment_sha256=env_hash,
        )
        verify_retrieval_index_consistency(
            staging_dir,
            manifest,
            label_order=labels,
        )
        publish_retrieval_baseline(staging_dir, final_dir)
    except Exception:
        best_effort_remove_tree(staging_dir)
        raise
    else:
        if staging_dir.exists():
            best_effort_remove_tree(staging_dir)

    return RetrievalBuildResult(
        repository=repository,
        retrieval_dir=final_dir,
        manifest=manifest,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        cache_hit=False,
    )


def format_retrieval_summary(result: RetrievalBuildResult) -> str:
    manifest = result.manifest
    lines = [
        f"repository: {result.repository.full_name}",
        f"model_dataset_id: {manifest.model_dataset_id}",
        f"retrieval_run_id: {manifest.retrieval_run_id}",
        f"corpus_size: {manifest.corpus_size}",
        f"validation_query_count: {manifest.validation_query_count}",
        f"test_query_count: {manifest.test_query_count}",
        f"top_k: {manifest.top_k}",
        f"validation_recall_at_5: {result.validation_metrics.recall_at_5:.6f}",
        f"validation_recall_at_10: {result.validation_metrics.recall_at_10:.6f}",
        f"test_recall_at_5: {result.test_metrics.recall_at_5:.6f}",
        f"test_recall_at_10: {result.test_metrics.recall_at_10:.6f}",
        f"artifact_path: {result.retrieval_dir}",
        f"cache_hit: {str(result.cache_hit).lower()}",
    ]
    return "\n".join(lines)
