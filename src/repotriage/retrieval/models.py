"""Pydantic models, identity hashing, and domain exceptions for retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from repotriage.baseline.models import TfidfParams
from repotriage.dataset.models import Sha256Hex
from repotriage.model_dataset.models import SplitName

RETRIEVAL_BASELINE_VERSION: Literal["1"] = "1"
RETRIEVAL_MANIFEST_SCHEMA_VERSION: Literal["1"] = "1"
RETRIEVAL_CONFIG_SCHEMA_VERSION: Literal["1"] = "1"
RETRIEVAL_PROTOCOL_VERSION: Literal["train_corpus_v1"] = "train_corpus_v1"
RETRIEVAL_METRIC_CONTRACT_VERSION: Literal["1"] = "1"
INDEX_SEMANTIC_CONTRACT_VERSION: Literal["1"] = "1"

CONFIG_JSON_FILE = "config.json"
INDEX_METADATA_JSON_FILE = "index_metadata.json"
CORPUS_RECORDS_JSONL_FILE = "corpus_records.jsonl"
NEIGHBORS_VALIDATION_JSONL_FILE = "neighbors_validation.jsonl"
NEIGHBORS_TEST_JSONL_FILE = "neighbors_test.jsonl"
METRICS_VALIDATION_JSON_FILE = "metrics_validation.json"
METRICS_TEST_JSON_FILE = "metrics_test.json"
REPORT_MARKDOWN_FILE = "report.md"
MANIFEST_JSON_FILE = "manifest.json"
VECTORIZER_JOBLIB_FILE = "vectorizer.joblib"
CORPUS_MATRIX_NPZ_FILE = "corpus_matrix.npz"

RETRIEVAL_RUN_ID_PATTERN = (
    r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-md[1-9]\d*-[0-9a-f]{12}"
    r"-rb[1-9]\d*-[0-9a-f]{12}$"
)
_RETRIEVAL_RUN_ID_RE = re.compile(RETRIEVAL_RUN_ID_PATTERN)

RetrievalRunId = Annotated[str, StringConstraints(pattern=RETRIEVAL_RUN_ID_PATTERN)]

SimilarityMetric = Literal["cosine"]
RelevanceDefinition = Literal["label_overlap"]

_FLOAT_ABS_TOL = 1e-12


def floats_consistent(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)


class RetrievalError(RuntimeError):
    """Base class for retrieval domain errors."""


class RetrievalConfigError(RetrievalError):
    """Raised when the human-authored configuration is missing, invalid, or inconsistent."""


class RetrievalInputError(RetrievalError):
    """Raised when model-ready inputs are missing or incompatible."""


class RetrievalBuildError(RetrievalError):
    """Raised when artifact publication fails."""


class RetrievalCorruptionError(RetrievalError):
    """Raised when an on-disk artifact fails integrity validation."""


class RetrievalConfigDocument(BaseModel):
    """Human-authored retrieval-baseline configuration."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["1"] = RETRIEVAL_CONFIG_SCHEMA_VERSION
    retrieval_baseline_version: Literal["1"] = RETRIEVAL_BASELINE_VERSION
    repository: str = Field(min_length=1)
    retrieval_protocol_version: Literal["train_corpus_v1"] = RETRIEVAL_PROTOCOL_VERSION
    metric_contract_version: Literal["1"] = RETRIEVAL_METRIC_CONTRACT_VERSION
    similarity_metric: SimilarityMetric = "cosine"
    relevance_definition: RelevanceDefinition = "label_overlap"
    top_k: int = Field(ge=1)
    tfidf: TfidfParams


class FrozenRetrievalConfig(BaseModel):
    """Published semantic configuration snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    retrieval_baseline_version: Literal["1"] = RETRIEVAL_BASELINE_VERSION
    repository: str
    retrieval_protocol_version: Literal["train_corpus_v1"] = RETRIEVAL_PROTOCOL_VERSION
    metric_contract_version: Literal["1"] = RETRIEVAL_METRIC_CONTRACT_VERSION
    similarity_metric: SimilarityMetric
    relevance_definition: RelevanceDefinition
    top_k: int = Field(ge=1)
    tfidf: TfidfParams


class CorpusRecord(BaseModel):
    """One train-corpus row (metadata only)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    issue_id: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    created_at: str
    selected_labels: list[str] = Field(default_factory=list)
    corpus_index: int = Field(ge=0)


class NeighborRecord(BaseModel):
    """One retrieved neighbor for a query."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    neighbor_issue_id: int = Field(gt=0)
    neighbor_issue_number: int = Field(gt=0)
    similarity: float = Field(ge=-1.0, le=1.0)
    neighbor_selected_labels: list[str] = Field(default_factory=list)
    shared_labels: list[str] = Field(default_factory=list)
    label_jaccard: float = Field(ge=0.0, le=1.0)
    is_relevant: bool


class QueryNeighborRecord(BaseModel):
    """Retrieval results for one query issue."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    query_issue_id: int = Field(gt=0)
    query_issue_number: int = Field(gt=0)
    query_split: SplitName
    query_selected_labels: list[str] = Field(default_factory=list)
    query_has_positive_labels: bool
    neighbors: list[NeighborRecord]


class RetrievalMetricsDocument(BaseModel):
    """Aggregate label-overlap retrieval metrics for one query split."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    metric_contract_version: Literal["1"] = RETRIEVAL_METRIC_CONTRACT_VERSION
    split: SplitName
    total_query_count: int = Field(ge=0)
    scored_query_count: int = Field(ge=0)
    all_zero_label_query_count: int = Field(ge=0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    precision_at_5: float = Field(ge=0.0, le=1.0)
    precision_at_10: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    mean_best_label_jaccard_at_10: float = Field(ge=0.0, le=1.0)
    mean_best_shared_label_count_at_10: float = Field(ge=0.0)


class IndexMetadataDocument(BaseModel):
    """Summary of the fitted retrieval index."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    index_semantic_contract_version: Literal["1"] = INDEX_SEMANTIC_CONTRACT_VERSION
    index_semantic_sha256: Sha256Hex
    retrieval_baseline_version: Literal["1"] = RETRIEVAL_BASELINE_VERSION
    model_dataset_id: str
    repository: str
    corpus_size: int = Field(ge=0)
    vocabulary_size: int = Field(ge=0)
    matrix_shape: tuple[int, int]
    train_issue_ids: list[int]
    top_k: int = Field(ge=1)
    similarity_metric: SimilarityMetric
    metric_contract_version: Literal["1"] = RETRIEVAL_METRIC_CONTRACT_VERSION


class RetrievalManifest(BaseModel):
    """Validated lineage manifest for one immutable retrieval-baseline artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = RETRIEVAL_MANIFEST_SCHEMA_VERSION
    retrieval_baseline_version: Literal["1"] = RETRIEVAL_BASELINE_VERSION
    retrieval_run_id: RetrievalRunId
    retrieval_experiment_sha256: Sha256Hex
    numerical_environment_sha256: Sha256Hex
    retrieval_run_sha256: Sha256Hex
    config_source_sha256: Sha256Hex
    config_semantic_sha256: Sha256Hex
    repository: str
    model_dataset_id: str
    records_sha256: Sha256Hex
    label_map_sha256: Sha256Hex
    retrieval_protocol_version: Literal["train_corpus_v1"] = RETRIEVAL_PROTOCOL_VERSION
    metric_contract_version: Literal["1"] = RETRIEVAL_METRIC_CONTRACT_VERSION
    similarity_metric: SimilarityMetric
    relevance_definition: RelevanceDefinition
    top_k: int = Field(ge=1)
    corpus_size: int = Field(ge=0)
    validation_query_count: int = Field(ge=0)
    test_query_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    train_issue_ids: list[int]
    index_semantic_sha256: Sha256Hex
    index_semantic_contract_version: Literal["1"] = INDEX_SEMANTIC_CONTRACT_VERSION
    built_at: str
    config_file: str = CONFIG_JSON_FILE
    config_sha256: Sha256Hex
    index_metadata_file: str = INDEX_METADATA_JSON_FILE
    index_metadata_sha256: Sha256Hex
    corpus_records_file: str = CORPUS_RECORDS_JSONL_FILE
    corpus_records_sha256: Sha256Hex
    neighbors_validation_file: str = NEIGHBORS_VALIDATION_JSONL_FILE
    neighbors_validation_sha256: Sha256Hex
    neighbors_test_file: str = NEIGHBORS_TEST_JSONL_FILE
    neighbors_test_sha256: Sha256Hex
    metrics_validation_file: str = METRICS_VALIDATION_JSON_FILE
    metrics_validation_sha256: Sha256Hex
    metrics_test_file: str = METRICS_TEST_JSON_FILE
    metrics_test_sha256: Sha256Hex
    report_file: str = REPORT_MARKDOWN_FILE
    report_sha256: Sha256Hex
    vectorizer_file: str = VECTORIZER_JOBLIB_FILE
    vectorizer_sha256: Sha256Hex
    corpus_matrix_file: str = CORPUS_MATRIX_NPZ_FILE
    corpus_matrix_sha256: Sha256Hex


def validate_retrieval_run_id(value: str) -> str:
    if not _RETRIEVAL_RUN_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid retrieval_run_id: {value!r}")
    return value


def compute_retrieval_experiment_sha256(
    *,
    retrieval_baseline_version: str,
    retrieval_protocol_version: str,
    metric_contract_version: str,
    model_dataset_id: str,
    records_sha256: str,
    label_map_sha256: str,
    config_semantic_sha256: str,
    top_k: int,
    similarity_metric: str,
    relevance_definition: str,
    tfidf: TfidfParams,
    label_order: list[str],
) -> str:
    """Derive the semantic experiment hash binding retrieval intent."""
    payload = {
        "config_semantic_sha256": config_semantic_sha256,
        "label_map_sha256": label_map_sha256,
        "label_order": label_order,
        "metric_contract_version": metric_contract_version,
        "model_dataset_id": model_dataset_id,
        "records_sha256": records_sha256,
        "relevance_definition": relevance_definition,
        "retrieval_baseline_version": retrieval_baseline_version,
        "retrieval_protocol_version": retrieval_protocol_version,
        "similarity_metric": similarity_metric,
        "tfidf": tfidf.model_dump(mode="json"),
        "top_k": top_k,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_retrieval_run_sha256(
    retrieval_experiment_sha256: str,
    numerical_environment_sha256: str,
) -> str:
    """Combine experiment and environment hashes into one run identity."""
    payload = {
        "numerical_environment_sha256": numerical_environment_sha256,
        "retrieval_experiment_sha256": retrieval_experiment_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_retrieval_run_id(
    model_dataset_id: str,
    retrieval_run_sha256: str,
    retrieval_baseline_version: str = RETRIEVAL_BASELINE_VERSION,
) -> str:
    """Derive a content-aware retrieval run id from model-dataset id and run hash."""
    short_hash = retrieval_run_sha256[:12]
    return f"{model_dataset_id}-rb{retrieval_baseline_version}-{short_hash}"
