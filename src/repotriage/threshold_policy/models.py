"""Pydantic models, identity hashing, and domain exceptions for threshold-policy artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from repotriage.baseline.models import METRIC_CONTRACT_VERSION, SplitMetrics
from repotriage.dataset.models import Sha256Hex

THRESHOLD_POLICY_VERSION: Literal["1"] = "1"
THRESHOLD_POLICY_CONFIG_SCHEMA_VERSION: Literal["1"] = "1"
THRESHOLD_POLICY_MANIFEST_SCHEMA_VERSION: Literal["1"] = "1"
THRESHOLD_SELECTION_RULE_VERSION: Literal["1"] = "1"

CONFIG_JSON_FILE = "config.json"
POLICY_JSON_FILE = "policy.json"
SWEEP_VALIDATION_JSON_FILE = "sweep_validation.json"
METRICS_VALIDATION_JSON_FILE = "metrics_validation.json"
METRICS_TEST_JSON_FILE = "metrics_test.json"
COMPARISON_JSON_FILE = "comparison.json"
REPORT_MARKDOWN_FILE = "report.md"
MANIFEST_JSON_FILE = "manifest.json"

POLICY_ID_PATTERN = (
    r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-md[1-9]\d*-[0-9a-f]{12}"
    r"-bl[1-9]\d*-[0-9a-f]{12}-tp[1-9]\d*-[0-9a-f]{12}$"
)
_POLICY_ID_RE = re.compile(POLICY_ID_PATTERN)

PolicyId = Annotated[str, StringConstraints(pattern=POLICY_ID_PATTERN)]


class ThresholdPolicyError(RuntimeError):
    """Base class for threshold-policy domain errors."""


class ThresholdPolicyConfigError(ThresholdPolicyError):
    """Raised when the human-authored configuration is missing, invalid, or inconsistent."""


class ThresholdPolicyInputError(ThresholdPolicyError):
    """Raised when baseline inputs are missing or incompatible."""


class ThresholdPolicyBuildError(ThresholdPolicyError):
    """Raised when artifact publication fails."""


class ThresholdPolicyCorruptionError(ThresholdPolicyError):
    """Raised when an on-disk artifact fails integrity validation."""


class ThresholdGridConfig(BaseModel):
    """Integer basis-point threshold grid definition."""

    model_config = ConfigDict(extra="forbid")

    start_basis_points: int = Field(ge=0)
    stop_basis_points: int = Field(ge=0)
    step_basis_points: int = Field(gt=0)
    denominator: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_grid(self) -> ThresholdGridConfig:
        if self.stop_basis_points < self.start_basis_points:
            raise ValueError("stop_basis_points must be >= start_basis_points")
        span = self.stop_basis_points - self.start_basis_points
        if span % self.step_basis_points != 0:
            raise ValueError("grid span must be divisible by step_basis_points")
        return self

    def basis_points(self) -> list[int]:
        return list(
            range(
                self.start_basis_points,
                self.stop_basis_points + 1,
                self.step_basis_points,
            )
        )

    @staticmethod
    def threshold_from_basis_points(basis_points: int, denominator: int) -> float:
        return basis_points / denominator


class ThresholdPolicyConfigDocument(BaseModel):
    """Human-authored threshold-policy configuration."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["1"] = THRESHOLD_POLICY_CONFIG_SCHEMA_VERSION
    threshold_policy_version: Literal["1"] = THRESHOLD_POLICY_VERSION
    repository: str = Field(min_length=1)
    baseline_run_id: str = Field(min_length=1)
    selected_candidate_id: str = Field(min_length=1)
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    selection_rule_version: Literal["1"] = THRESHOLD_SELECTION_RULE_VERSION
    reference_threshold_basis_points: int = Field(ge=0)
    threshold_grid: ThresholdGridConfig


class FrozenThresholdPolicyConfig(BaseModel):
    """Published semantic configuration snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    threshold_policy_version: Literal["1"] = THRESHOLD_POLICY_VERSION
    repository: str
    baseline_run_id: str
    selected_candidate_id: str
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    selection_rule_version: Literal["1"] = THRESHOLD_SELECTION_RULE_VERSION
    reference_threshold_basis_points: int = Field(ge=0)
    threshold_grid: ThresholdGridConfig


class ThresholdSelectionAudit(BaseModel):
    """Record of how the global threshold was chosen on validation."""

    model_config = ConfigDict(extra="forbid")

    selection_rule_version: Literal["1"] = THRESHOLD_SELECTION_RULE_VERSION
    selected_threshold_basis_points: int = Field(ge=0)
    reference_threshold_basis_points: int = Field(ge=0)
    tie_break_steps: list[str] = Field(default_factory=list)
    ranked_threshold_basis_points: list[int] = Field(default_factory=list)


class PolicyDocument(BaseModel):
    """Selected threshold decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    selection: ThresholdSelectionAudit


class ThresholdSweepRow(BaseModel):
    """Validation metrics at one grid threshold."""

    model_config = ConfigDict(extra="forbid")

    threshold_basis_points: int = Field(ge=0)
    threshold: float = Field(ge=0.0, le=1.0)
    metrics: SplitMetrics


class SweepValidationDocument(BaseModel):
    """Full validation threshold sweep."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    threshold_grid: ThresholdGridConfig
    rows: list[ThresholdSweepRow]


class ComparisonSplitMetrics(BaseModel):
    """Aggregate metrics for one threshold on one split."""

    model_config = ConfigDict(extra="forbid")

    threshold_basis_points: int = Field(ge=0)
    threshold: float = Field(ge=0.0, le=1.0)
    macro_f1: float | None = None
    micro_f1: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    subset_accuracy: float = Field(ge=0.0, le=1.0)
    hamming_loss: float = Field(ge=0.0, le=1.0)
    mean_true_label_cardinality: float = Field(ge=0.0)
    mean_predicted_label_cardinality: float = Field(ge=0.0)
    fraction_no_prediction: float = Field(ge=0.0, le=1.0)
    fraction_any_prediction: float = Field(ge=0.0, le=1.0)
    predicted_positives_by_label: dict[str, int] = Field(default_factory=dict)


class ComparisonSplitPair(BaseModel):
    """Reference vs selected metrics on one split."""

    model_config = ConfigDict(extra="forbid")

    reference: ComparisonSplitMetrics
    selected: ComparisonSplitMetrics


class ComparisonDocument(BaseModel):
    """Baseline reference threshold vs selected threshold."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    reference_threshold_basis_points: int = Field(ge=0)
    selected_threshold_basis_points: int = Field(ge=0)
    validation: ComparisonSplitPair
    test: ComparisonSplitPair


class ThresholdPolicyManifest(BaseModel):
    """Validated lineage manifest for one immutable threshold-policy artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = THRESHOLD_POLICY_MANIFEST_SCHEMA_VERSION
    threshold_policy_version: Literal["1"] = THRESHOLD_POLICY_VERSION
    policy_id: PolicyId
    policy_input_sha256: Sha256Hex
    config_source_sha256: Sha256Hex
    config_semantic_sha256: Sha256Hex
    repository: str
    baseline_run_id: str
    baseline_experiment_sha256: Sha256Hex
    model_dataset_id: str
    model_semantic_sha256: Sha256Hex
    selected_candidate_id: str
    predictions_validation_sha256: Sha256Hex
    predictions_test_sha256: Sha256Hex
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    selection_rule_version: Literal["1"] = THRESHOLD_SELECTION_RULE_VERSION
    reference_threshold_basis_points: int = Field(ge=0)
    selected_threshold_basis_points: int = Field(ge=0)
    validation_record_count: int = Field(ge=0)
    test_record_count: int = Field(ge=0)
    target_count: int = Field(ge=0)
    sweep_threshold_count: int = Field(ge=0)
    built_at: str
    config_file: str = CONFIG_JSON_FILE
    config_sha256: Sha256Hex
    policy_file: str = POLICY_JSON_FILE
    policy_sha256: Sha256Hex
    sweep_validation_file: str = SWEEP_VALIDATION_JSON_FILE
    sweep_validation_sha256: Sha256Hex
    metrics_validation_file: str = METRICS_VALIDATION_JSON_FILE
    metrics_validation_sha256: Sha256Hex
    metrics_test_file: str = METRICS_TEST_JSON_FILE
    metrics_test_sha256: Sha256Hex
    comparison_file: str = COMPARISON_JSON_FILE
    comparison_sha256: Sha256Hex
    report_file: str = REPORT_MARKDOWN_FILE
    report_sha256: Sha256Hex


def validate_policy_id(value: str) -> str:
    if not _POLICY_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid policy_id: {value!r}")
    return value


def compute_policy_input_sha256(
    *,
    threshold_policy_version: str,
    baseline_run_id: str,
    baseline_experiment_sha256: str,
    model_semantic_sha256: str,
    predictions_validation_sha256: str,
    predictions_test_sha256: str,
    selected_candidate_id: str,
    threshold_grid: ThresholdGridConfig,
    selection_rule_version: str,
    metric_contract_version: str,
) -> str:
    """Hash the canonical policy-input payload that fully determines the generated policy."""
    payload = {
        "baseline_experiment_sha256": baseline_experiment_sha256,
        "baseline_run_id": baseline_run_id,
        "metric_contract_version": metric_contract_version,
        "model_semantic_sha256": model_semantic_sha256,
        "predictions_test_sha256": predictions_test_sha256,
        "predictions_validation_sha256": predictions_validation_sha256,
        "selected_candidate_id": selected_candidate_id,
        "selection_rule_version": selection_rule_version,
        "threshold_grid": threshold_grid.model_dump(mode="json"),
        "threshold_policy_version": threshold_policy_version,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_policy_id(baseline_run_id: str, policy_input_sha256: str) -> str:
    """Derive the deterministic policy id from the policy-input hash."""
    return f"{baseline_run_id}-tp{THRESHOLD_POLICY_VERSION}-{policy_input_sha256[:12]}"


def comparison_split_metrics_from_split_metrics(
    metrics: SplitMetrics,
    *,
    threshold_basis_points: int,
    threshold: float,
) -> ComparisonSplitMetrics:
    aggregate = metrics.aggregate
    return ComparisonSplitMetrics(
        threshold_basis_points=threshold_basis_points,
        threshold=threshold,
        macro_f1=aggregate.macro_f1,
        micro_f1=aggregate.micro_f1,
        macro_precision=aggregate.macro_precision,
        macro_recall=aggregate.macro_recall,
        subset_accuracy=aggregate.subset_accuracy,
        hamming_loss=aggregate.hamming_loss,
        mean_true_label_cardinality=aggregate.mean_true_label_cardinality,
        mean_predicted_label_cardinality=aggregate.mean_predicted_label_cardinality,
        fraction_no_prediction=aggregate.fraction_no_prediction,
        fraction_any_prediction=aggregate.fraction_any_prediction,
        predicted_positives_by_label={
            item.label: item.predicted_positives for item in metrics.per_label
        },
    )
