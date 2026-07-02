"""Pydantic models, identity hashing, and domain exceptions for baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from repotriage.dataset.models import Sha256Hex
from repotriage.model_dataset.models import SplitName

BASELINE_VERSION: Literal["4"] = "4"
BASELINE_MANIFEST_SCHEMA_VERSION: Literal["4"] = "4"
MODEL_SEMANTIC_CONTRACT_VERSION: Literal["1"] = "1"
PREDICTION_RECORD_SCHEMA_VERSION: Literal["2"] = "2"
METRIC_CONTRACT_VERSION: Literal["2"] = "2"
ENVIRONMENT_SCHEMA_VERSION: Literal["2"] = "2"
SELECTION_RULE_VERSION: Literal["1"] = "1"
TRAINING_PROTOCOL_VERSION: Literal["train_only_v1"] = "train_only_v1"
CANDIDATE_SET_VERSION: Literal["1"] = "1"
SAMPLES_F1_EMPTY_EMPTY_POLICY: Literal["one"] = "one"
NUMERICAL_THREAD_LIMIT: int = 1

CONFIG_JSON_FILE = "config.json"
CANDIDATE_RESULTS_JSON_FILE = "candidate_results.json"
METRICS_TEST_JSON_FILE = "metrics_test.json"
METRICS_MARKDOWN_FILE = "metrics.md"
PREDICTIONS_VALIDATION_JSONL_FILE = "predictions_validation.jsonl"
PREDICTIONS_TEST_JSONL_FILE = "predictions_test.jsonl"
FEATURE_SUMMARY_JSON_FILE = "feature_summary.json"
MODEL_JOBLIB_FILE = "model.joblib"

BASELINE_RUN_ID_PATTERN = (
    r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-md[1-9]\d*-[0-9a-f]{12}-bl[1-9]\d*-[0-9a-f]{12}$"
)
_BASELINE_RUN_ID_RE = re.compile(BASELINE_RUN_ID_PATTERN)

BaselineRunId = Annotated[str, StringConstraints(pattern=BASELINE_RUN_ID_PATTERN)]

ScoreType = Literal["probability_estimates", "none"]
ClassWeight = Literal["balanced"] | None

_FLOAT_ABS_TOL = 1e-12


def floats_consistent(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)


class BaselineError(RuntimeError):
    """Base class for baseline domain errors."""


class BaselineConfigError(BaselineError):
    """Raised when a baseline configuration is invalid or incompatible."""


class BaselineInputError(BaselineError):
    """Raised when required model-ready inputs are missing or mismatched."""


class BaselineCorruptionError(BaselineError):
    """Raised when an existing baseline artifact is corrupt or incompatible."""


class BaselineTrainingError(BaselineError):
    """Raised when model training fails."""


class BaselineBuildError(BaselineError):
    """Raised when staging or publication of a baseline artifact fails."""


class TfidfParams(BaseModel):
    """TF-IDF vectorizer parameters for one baseline candidate."""

    model_config = ConfigDict(extra="forbid")

    analyzer: Literal["word"] = "word"
    ngram_range: tuple[int, int]
    lowercase: bool = True
    min_df: int = Field(ge=1)
    sublinear_tf: bool = True
    norm: Literal["l2"] = "l2"


class LogRegParams(BaseModel):
    """Logistic regression parameters for one baseline candidate."""

    model_config = ConfigDict(extra="forbid")

    C: float = Field(gt=0.0)
    solver: Literal["lbfgs"] = "lbfgs"
    max_iter: int = Field(ge=1)
    class_weight: ClassWeight = None


class BaselineCandidateConfig(BaseModel):
    """One predeclared baseline candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    tfidf: TfidfParams
    logreg: LogRegParams


class ThresholdPolicy(BaseModel):
    """Fixed threshold policy for binary decisions from continuous scores."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(ge=0.0, le=1.0)
    score_type: Literal["probability_estimates"] = "probability_estimates"


class BaselineConfigDocument(BaseModel):
    """Human-authored baseline training configuration."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["1"] = "1"
    baseline_version: Literal["4"] = BASELINE_VERSION
    repository: str
    candidate_set_version: Literal["1"] = CANDIDATE_SET_VERSION
    selection_rule_version: Literal["1"] = SELECTION_RULE_VERSION
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    training_protocol_version: Literal["train_only_v1"] = TRAINING_PROTOCOL_VERSION
    random_state: int
    threshold_policy: ThresholdPolicy
    candidates: list[BaselineCandidateConfig] = Field(min_length=1)


class PerLabelMetric(BaseModel):
    """Threshold-dependent and independent metrics for one label."""

    model_config = ConfigDict(extra="forbid")

    label: str
    support: int = Field(ge=0)
    prevalence: float = Field(ge=0.0, le=1.0)
    predicted_positives: int = Field(ge=0)
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    precision_undefined_reason: str | None = None
    recall_undefined_reason: str | None = None
    average_precision: float | None = None
    roc_auc: float | None = None
    roc_auc_undefined_reason: str | None = None


class AggregateMetrics(BaseModel):
    """Dataset-level multilabel metrics (metric contract v2)."""

    model_config = ConfigDict(extra="forbid")

    micro_precision: float | None = None
    micro_recall: float | None = None
    micro_f1: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    macro_precision_defined_only: float | None = None
    macro_recall_defined_only: float | None = None
    macro_f1_defined_only: float | None = None
    macro_label_count: int = Field(ge=0)
    macro_precision_denominator: int = Field(ge=0)
    macro_recall_denominator: int = Field(ge=0)
    macro_f1_denominator: int = Field(ge=0)
    undefined_precision_label_count: int = Field(ge=0)
    undefined_recall_label_count: int = Field(ge=0)
    undefined_f1_label_count: int = Field(ge=0)
    macro_labels_skipped_precision: list[str] = Field(default_factory=list)
    macro_labels_skipped_recall: list[str] = Field(default_factory=list)
    macro_labels_skipped_f1: list[str] = Field(default_factory=list)
    weighted_f1: float | None = None
    samples_f1: float | None = None
    samples_f1_empty_empty_policy: Literal["one"] = SAMPLES_F1_EMPTY_EMPTY_POLICY
    subset_accuracy: float = Field(ge=0.0, le=1.0)
    hamming_loss: float = Field(ge=0.0, le=1.0)
    macro_average_precision: float | None = None
    macro_average_precision_label_count: int = Field(ge=0)
    micro_average_precision: float | None = None
    mean_true_label_cardinality: float = Field(ge=0.0)
    mean_predicted_label_cardinality: float = Field(ge=0.0)
    fraction_no_prediction: float = Field(ge=0.0, le=1.0)
    fraction_any_prediction: float = Field(ge=0.0, le=1.0)
    record_count: int = Field(ge=0)


class SplitMetrics(BaseModel):
    """Full metric report for one evaluation split."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2"] = "2"
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    split: SplitName
    threshold: float
    score_type: ScoreType
    per_label: list[PerLabelMetric]
    aggregate: AggregateMetrics


class PredictionRecord(BaseModel):
    """One stored validation or test prediction."""

    schema_version: Literal["2"] = PREDICTION_RECORD_SCHEMA_VERSION
    candidate_id: str | None = None
    repository: str
    model_dataset_id: str
    baseline_run_id: str
    issue_id: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    split: SplitName
    true_labels: list[str]
    true_vector: list[int]
    predicted_labels: list[str]
    predicted_vector: list[int]
    score_type: ScoreType
    threshold: float | None = None
    score_vector: list[float] | None = None


class CandidateValidationResult(BaseModel):
    """Validation metrics and metadata for one candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    selectable: bool = True
    metrics: SplitMetrics
    vocabulary_size: int | None = None
    convergence_warnings: list[str] = Field(default_factory=list)
    label_convergence: dict[str, int] = Field(default_factory=dict)


class SelectionAudit(BaseModel):
    """Record of how the winning candidate was chosen."""

    model_config = ConfigDict(extra="forbid")

    selection_rule_version: Literal["1"] = SELECTION_RULE_VERSION
    winner_candidate_id: str
    tie_break_steps: list[str] = Field(default_factory=list)
    ranked_candidate_ids: list[str] = Field(default_factory=list)


class CandidateResultsDocument(BaseModel):
    """Validation results for all candidates plus selection audit."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2"] = "2"
    dummy_baseline: CandidateValidationResult | None = None
    candidates: list[CandidateValidationResult]
    selection: SelectionAudit


class FeatureSummary(BaseModel):
    """Post-fit feature and training diagnostics for the selected candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2"] = "2"
    train_only_fit: bool = True
    vocabulary_size: int = Field(ge=0)
    train_record_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    convergence_warnings: list[str] = Field(default_factory=list)
    label_convergence: dict[str, int] = Field(default_factory=dict)


class BackendModule(BaseModel):
    """One numerical backend module loaded at runtime (from threadpoolctl)."""

    model_config = ConfigDict(extra="forbid")

    user_api: str | None = None
    internal_api: str | None = None
    prefix: str | None = None
    version: str | None = None
    threading_layer: str | None = None
    architecture: str | None = None


class EnvironmentMetadata(BaseModel):
    """Runtime environment recorded for numerical reproducibility context."""

    model_config = ConfigDict(extra="forbid")

    environment_schema_version: Literal["2"] = ENVIRONMENT_SCHEMA_VERSION
    python_implementation: str
    python_version: str
    os_system: str
    platform: str
    machine_architecture: str
    numpy_version: str | None = None
    scipy_version: str | None = None
    scikit_learn_version: str | None = None
    joblib_version: str | None = None
    threadpoolctl_version: str | None = None
    blas_lapack_vendor: str | None = None
    numerical_backends: list[BackendModule] = Field(default_factory=list)
    numerical_thread_limit: int = Field(ge=1)
    reproducibility_note: str
    serialization_security_warning: str


class FrozenConfigDocument(BaseModel):
    """Published configuration for the selected frozen baseline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2"] = "2"
    baseline_version: Literal["4"] = BASELINE_VERSION
    repository: str
    model_dataset_id: str
    selected_candidate_id: str
    candidate_set_version: Literal["1"] = CANDIDATE_SET_VERSION
    selection_rule_version: Literal["1"] = SELECTION_RULE_VERSION
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    training_protocol_version: Literal["train_only_v1"] = TRAINING_PROTOCOL_VERSION
    random_state: int
    threshold_policy: ThresholdPolicy
    selected_candidate: BaselineCandidateConfig
    all_candidates: list[BaselineCandidateConfig]


class BaselineManifest(BaseModel):
    """Validated lineage manifest describing one immutable baseline artifact."""

    schema_version: Literal["4"] = BASELINE_MANIFEST_SCHEMA_VERSION
    baseline_version: Literal["4"] = BASELINE_VERSION
    baseline_run_id: BaselineRunId
    baseline_experiment_sha256: Sha256Hex
    numerical_environment_sha256: Sha256Hex
    baseline_run_sha256: Sha256Hex
    repository: str
    model_dataset_id: str
    records_sha256: Sha256Hex
    label_map_sha256: Sha256Hex
    config_semantic_sha256: Sha256Hex
    config_source_sha256: Sha256Hex
    candidate_set_version: Literal["1"] = CANDIDATE_SET_VERSION
    selection_rule_version: Literal["1"] = SELECTION_RULE_VERSION
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    training_protocol_version: Literal["train_only_v1"] = TRAINING_PROTOCOL_VERSION
    random_state: int
    threshold: float
    score_type: Literal["probability_estimates"] = "probability_estimates"
    selected_candidate_id: str
    built_at: str
    validation_record_count: int = Field(ge=0)
    validation_prediction_count: int = Field(ge=0)
    test_record_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    config_file: str = CONFIG_JSON_FILE
    config_sha256: Sha256Hex
    candidate_results_file: str = CANDIDATE_RESULTS_JSON_FILE
    candidate_results_sha256: Sha256Hex
    metrics_test_file: str = METRICS_TEST_JSON_FILE
    metrics_test_sha256: Sha256Hex
    metrics_markdown_file: str = METRICS_MARKDOWN_FILE
    metrics_markdown_sha256: Sha256Hex
    predictions_validation_file: str = PREDICTIONS_VALIDATION_JSONL_FILE
    predictions_validation_sha256: Sha256Hex
    predictions_test_file: str = PREDICTIONS_TEST_JSONL_FILE
    predictions_test_sha256: Sha256Hex
    feature_summary_file: str = FEATURE_SUMMARY_JSON_FILE
    feature_summary_sha256: Sha256Hex
    model_file: str = MODEL_JOBLIB_FILE
    model_sha256: Sha256Hex
    model_semantic_sha256: Sha256Hex
    model_semantic_contract_version: Literal["1"] = MODEL_SEMANTIC_CONTRACT_VERSION
    environment: EnvironmentMetadata


def compute_baseline_experiment_sha256(
    *,
    baseline_version: str,
    model_dataset_id: str,
    records_sha256: str,
    label_map_sha256: str,
    config_semantic_sha256: str,
    candidate_set_version: str,
    selection_rule_version: str,
    metric_contract_version: str,
    model_semantic_contract_version: str,
    threshold: float,
    score_type: str,
    training_protocol_version: str,
    random_state: int,
) -> str:
    """Derive the semantic experiment hash binding modeling intent."""
    payload = {
        "baseline_version": baseline_version,
        "candidate_set_version": candidate_set_version,
        "config_semantic_sha256": config_semantic_sha256,
        "label_map_sha256": label_map_sha256,
        "metric_contract_version": metric_contract_version,
        "model_dataset_id": model_dataset_id,
        "model_semantic_contract_version": model_semantic_contract_version,
        "random_state": random_state,
        "records_sha256": records_sha256,
        "score_type": score_type,
        "selection_rule_version": selection_rule_version,
        "threshold": threshold,
        "training_protocol_version": training_protocol_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_backend_entries(backends: list[BackendModule]) -> list[dict]:
    """Reduce backend modules to stable, output-relevant fields, sorted deterministically."""
    entries = [
        {
            "architecture": backend.architecture,
            "internal_api": backend.internal_api,
            "prefix": backend.prefix,
            "threading_layer": backend.threading_layer,
            "user_api": backend.user_api,
            "version": backend.version,
        }
        for backend in backends
    ]
    entries.sort(
        key=lambda entry: (
            entry["user_api"] or "",
            entry["internal_api"] or "",
            entry["prefix"] or "",
            entry["version"] or "",
            entry["threading_layer"] or "",
            entry["architecture"] or "",
        )
    )
    return entries


def compute_numerical_environment_sha256(
    *,
    environment_schema_version: str,
    python_implementation: str,
    python_version: str,
    os_system: str,
    machine_architecture: str,
    numpy_version: str | None,
    scipy_version: str | None,
    scikit_learn_version: str | None,
    joblib_version: str | None,
    threadpoolctl_version: str | None,
    blas_lapack_vendor: str | None,
    numerical_backends: list[BackendModule],
    numerical_thread_limit: int,
) -> str:
    """Derive a canonical numerical-environment fingerprint.

    Binds interpreter, OS, package versions, the structured numerical-backend
    fingerprint (including exact BLAS/LAPACK backend versions and threading layer),
    and the controlled numerical thread limit. Volatile fields (absolute file paths,
    installation directories, hostnames, PIDs, live thread counts) are excluded.
    """
    payload = {
        "blas_lapack_vendor": blas_lapack_vendor,
        "environment_schema_version": environment_schema_version,
        "joblib_version": joblib_version,
        "machine_architecture": machine_architecture,
        "numerical_backends": canonical_backend_entries(numerical_backends),
        "numerical_thread_limit": numerical_thread_limit,
        "numpy_version": numpy_version,
        "os_system": os_system,
        "python_implementation": python_implementation,
        "python_version": python_version,
        "scikit_learn_version": scikit_learn_version,
        "scipy_version": scipy_version,
        "threadpoolctl_version": threadpoolctl_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_baseline_run_sha256(
    baseline_experiment_sha256: str,
    numerical_environment_sha256: str,
) -> str:
    """Combine experiment and environment hashes into one run identity."""
    payload = {
        "baseline_experiment_sha256": baseline_experiment_sha256,
        "numerical_environment_sha256": numerical_environment_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_baseline_run_id(
    model_dataset_id: str,
    baseline_run_sha256: str,
    baseline_version: str = BASELINE_VERSION,
) -> str:
    """Derive a content-aware baseline run id from model-dataset id and run hash."""
    short_hash = baseline_run_sha256[:12]
    return f"{model_dataset_id}-bl{baseline_version}-{short_hash}"


def validate_baseline_run_id(value: str) -> str:
    if not _BASELINE_RUN_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid baseline run id: {value!r}")
    return value
