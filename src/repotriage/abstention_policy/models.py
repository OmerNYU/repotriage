"""Pydantic models, identity hashing, and domain exceptions for abstention-policy artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from repotriage.baseline.models import METRIC_CONTRACT_VERSION, SplitMetrics
from repotriage.dataset.models import Sha256Hex
from repotriage.threshold_policy.models import ThresholdGridConfig

ABSTENTION_POLICY_VERSION: Literal["1"] = "1"
ABSTENTION_POLICY_CONFIG_SCHEMA_VERSION: Literal["1"] = "1"
ABSTENTION_POLICY_MANIFEST_SCHEMA_VERSION: Literal["1"] = "1"
ABSTENTION_SELECTION_RULE_VERSION: Literal["1"] = "1"
CONFIDENCE_DEFINITION_MAX_PREDICTED: Literal["max_predicted_label_score"] = (
    "max_predicted_label_score"
)

CONFIG_JSON_FILE = "config.json"
POLICY_JSON_FILE = "policy.json"
SWEEP_VALIDATION_JSON_FILE = "sweep_validation.json"
METRICS_VALIDATION_JSON_FILE = "metrics_validation.json"
METRICS_TEST_JSON_FILE = "metrics_test.json"
CONFIDENCE_BINS_VALIDATION_JSON_FILE = "confidence_bins_validation.json"
CONFIDENCE_BINS_TEST_JSON_FILE = "confidence_bins_test.json"
COMPARISON_JSON_FILE = "comparison.json"
REPORT_MARKDOWN_FILE = "report.md"
MANIFEST_JSON_FILE = "manifest.json"

POLICY_ID_PATTERN = (
    r"^\d{8}T\d{12}Z-n[1-9]\d*-[0-9a-f]{12}-md[1-9]\d*-[0-9a-f]{12}"
    r"-bl[1-9]\d*-[0-9a-f]{12}-tp[1-9]\d*-[0-9a-f]{12}"
    r"-ap[1-9]\d*-[0-9a-f]{12}$"
)
_POLICY_ID_RE = re.compile(POLICY_ID_PATTERN)

PolicyId = Annotated[str, StringConstraints(pattern=POLICY_ID_PATTERN)]


class AbstentionPolicyError(RuntimeError):
    """Base class for abstention-policy domain errors."""


class AbstentionPolicyConfigError(AbstentionPolicyError):
    """Raised when the human-authored configuration is missing, invalid, or inconsistent."""


class AbstentionPolicyInputError(AbstentionPolicyError):
    """Raised when upstream inputs are missing or incompatible."""


class AbstentionPolicyBuildError(AbstentionPolicyError):
    """Raised when artifact publication fails."""


class AbstentionPolicyCorruptionError(AbstentionPolicyError):
    """Raised when an on-disk artifact fails integrity validation."""


class AbstentionGridConfigDocument(BaseModel):
    """Human-authored abstention grid definition."""

    model_config = ConfigDict(extra="forbid")

    start_basis_points_override: int | None = Field(default=None, ge=0)
    stop_basis_points: int = Field(ge=0)
    step_basis_points: int = Field(gt=0)
    denominator: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_grid(self) -> AbstentionGridConfigDocument:
        start = self.start_basis_points_override
        if start is not None and self.stop_basis_points < start:
            raise ValueError("stop_basis_points must be >= start_basis_points_override")
        span = self.stop_basis_points - (start if start is not None else 0)
        if start is not None and span % self.step_basis_points != 0:
            raise ValueError("grid span must be divisible by step_basis_points")
        return self

    def resolve_grid(self, *, classification_threshold_basis_points: int) -> ThresholdGridConfig:
        start = (
            self.start_basis_points_override
            if self.start_basis_points_override is not None
            else classification_threshold_basis_points
        )
        if self.stop_basis_points < start:
            raise ValueError("stop_basis_points must be >= classification threshold")
        return ThresholdGridConfig(
            start_basis_points=start,
            stop_basis_points=self.stop_basis_points,
            step_basis_points=self.step_basis_points,
            denominator=self.denominator,
        )


class AbstentionPolicyConfigDocument(BaseModel):
    """Human-authored abstention-policy configuration."""

    model_config = ConfigDict(extra="forbid")

    config_schema_version: Literal["1"] = ABSTENTION_POLICY_CONFIG_SCHEMA_VERSION
    abstention_policy_version: Literal["1"] = ABSTENTION_POLICY_VERSION
    repository: str = Field(min_length=1)
    threshold_policy_id: str = Field(min_length=1)
    confidence_definition: Literal["max_predicted_label_score"] = (
        CONFIDENCE_DEFINITION_MAX_PREDICTED
    )
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    selection_rule_version: Literal["1"] = ABSTENTION_SELECTION_RULE_VERSION
    minimum_coverage: float = Field(ge=0.0, le=1.0)
    abstention_grid: AbstentionGridConfigDocument


class FrozenAbstentionPolicyConfig(BaseModel):
    """Published semantic configuration snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    abstention_policy_version: Literal["1"] = ABSTENTION_POLICY_VERSION
    repository: str
    threshold_policy_id: str
    baseline_run_id: str
    confidence_definition: Literal["max_predicted_label_score"] = (
        CONFIDENCE_DEFINITION_MAX_PREDICTED
    )
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    selection_rule_version: Literal["1"] = ABSTENTION_SELECTION_RULE_VERSION
    minimum_coverage: float = Field(ge=0.0, le=1.0)
    classification_threshold_basis_points: int = Field(ge=0)
    abstention_grid: ThresholdGridConfig


class HandledMetrics(BaseModel):
    """Quality metrics computed on handled issues only."""

    model_config = ConfigDict(extra="forbid")

    subset_accuracy: float | None = None
    samples_f1: float | None = None
    micro_precision: float | None = None
    micro_recall: float | None = None
    micro_f1: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    mean_predicted_label_cardinality: float | None = None
    mean_true_label_cardinality: float | None = None
    false_positive_count: int | None = None
    false_negative_count: int | None = None


class AbstentionSweepRow(BaseModel):
    """Validation diagnostics at one abstention threshold."""

    model_config = ConfigDict(extra="forbid")

    abstention_basis_points: int = Field(ge=0)
    abstention_threshold: float = Field(ge=0.0, le=1.0)
    total_count: int = Field(ge=0)
    handled_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    forced_abstention_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    handled_metrics: HandledMetrics


class SweepValidationDocument(BaseModel):
    """Full validation abstention sweep."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    classification_threshold_basis_points: int = Field(ge=0)
    abstention_grid: ThresholdGridConfig
    rows: list[AbstentionSweepRow]


class AbstentionSelectionAudit(BaseModel):
    """Record of how the abstention threshold was chosen on validation."""

    model_config = ConfigDict(extra="forbid")

    selection_rule_version: Literal["1"] = ABSTENTION_SELECTION_RULE_VERSION
    classification_threshold_basis_points: int = Field(ge=0)
    selected_abstention_basis_points: int = Field(ge=0)
    minimum_coverage: float = Field(ge=0.0, le=1.0)
    tie_break_steps: list[str] = Field(default_factory=list)
    ranked_abstention_basis_points: list[int] = Field(default_factory=list)


class PolicyDocument(BaseModel):
    """Selected abstention decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    selection: AbstentionSelectionAudit


class AbstentionSplitMetrics(BaseModel):
    """Handled-subset metrics and coverage for one split at the selected abstention threshold."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    split: Literal["validation", "test"]
    classification_threshold: float = Field(ge=0.0, le=1.0)
    abstention_threshold: float = Field(ge=0.0, le=1.0)
    total_count: int = Field(ge=0)
    handled_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    forced_abstention_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    handled_metrics: HandledMetrics
    full_set_reference: HandledMetrics | None = None


class ConfidenceBinRow(BaseModel):
    """Diagnostics for one confidence bin."""

    model_config = ConfigDict(extra="forbid")

    bin_label: str
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    issue_count: int = Field(ge=0)
    fraction_of_all_issues: float = Field(ge=0.0, le=1.0)
    subset_accuracy: float | None = None
    samples_f1: float | None = None
    mean_predicted_label_cardinality: float | None = None
    mean_true_label_cardinality: float | None = None


class NoPredictionBucket(BaseModel):
    """Issues with no predicted labels at the classification threshold."""

    model_config = ConfigDict(extra="forbid")

    issue_count: int = Field(ge=0)
    fraction_of_all_issues: float = Field(ge=0.0, le=1.0)


class ConfidenceBinsDocument(BaseModel):
    """Confidence-bin diagnostics for one split."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    split: Literal["validation", "test"]
    classification_threshold: float = Field(ge=0.0, le=1.0)
    confidence_definition: Literal["max_predicted_label_score"] = (
        CONFIDENCE_DEFINITION_MAX_PREDICTED
    )
    bins: list[ConfidenceBinRow]
    no_prediction_bucket: NoPredictionBucket


class ComparisonSplitMetrics(BaseModel):
    """Metrics for one comparison arm on one split."""

    model_config = ConfigDict(extra="forbid")

    abstention_basis_points: int | None = None
    abstention_threshold: float | None = None
    coverage: float = Field(ge=0.0, le=1.0)
    handled_count: int = Field(ge=0)
    subset_accuracy: float | None = None
    samples_f1: float | None = None
    micro_f1: float | None = None
    macro_f1: float | None = None
    mean_predicted_label_cardinality: float | None = None
    mean_true_label_cardinality: float | None = None
    false_positive_count: int | None = None
    false_negative_count: int | None = None


class ComparisonSplitPair(BaseModel):
    """Full-set classification-threshold reference vs selected abstention handled subset."""

    model_config = ConfigDict(extra="forbid")

    classification_threshold_full_set: ComparisonSplitMetrics
    selected_abstention_handled: ComparisonSplitMetrics


class ComparisonDocument(BaseModel):
    """Classification-threshold full set vs selected abstention handled subset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    classification_threshold_basis_points: int = Field(ge=0)
    selected_abstention_basis_points: int = Field(ge=0)
    validation: ComparisonSplitPair
    test: ComparisonSplitPair


class AbstentionPolicyManifest(BaseModel):
    """Validated lineage manifest for one immutable abstention-policy artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = ABSTENTION_POLICY_MANIFEST_SCHEMA_VERSION
    abstention_policy_version: Literal["1"] = ABSTENTION_POLICY_VERSION
    policy_id: PolicyId
    policy_input_sha256: Sha256Hex
    config_source_sha256: Sha256Hex
    config_semantic_sha256: Sha256Hex
    repository: str
    baseline_run_id: str
    baseline_experiment_sha256: Sha256Hex
    model_dataset_id: str
    model_semantic_sha256: Sha256Hex
    threshold_policy_id: str
    threshold_policy_sha256: Sha256Hex
    selected_candidate_id: str
    predictions_validation_sha256: Sha256Hex
    predictions_test_sha256: Sha256Hex
    confidence_definition: Literal["max_predicted_label_score"] = (
        CONFIDENCE_DEFINITION_MAX_PREDICTED
    )
    metric_contract_version: Literal["2"] = METRIC_CONTRACT_VERSION
    selection_rule_version: Literal["1"] = ABSTENTION_SELECTION_RULE_VERSION
    classification_threshold_basis_points: int = Field(ge=0)
    selected_abstention_basis_points: int = Field(ge=0)
    minimum_coverage: float = Field(ge=0.0, le=1.0)
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
    confidence_bins_validation_file: str = CONFIDENCE_BINS_VALIDATION_JSON_FILE
    confidence_bins_validation_sha256: Sha256Hex
    confidence_bins_test_file: str = CONFIDENCE_BINS_TEST_JSON_FILE
    confidence_bins_test_sha256: Sha256Hex
    comparison_file: str = COMPARISON_JSON_FILE
    comparison_sha256: Sha256Hex
    report_file: str = REPORT_MARKDOWN_FILE
    report_sha256: Sha256Hex


def validate_policy_id(value: str) -> str:
    if not _POLICY_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid policy_id: {value!r}")
    return value


def identity_abstention_grid_payload(grid: ThresholdGridConfig) -> dict[str, int]:
    return {
        "denominator": grid.denominator,
        "start": grid.start_basis_points,
        "step": grid.step_basis_points,
        "stop": grid.stop_basis_points,
    }


def compute_policy_input_sha256(
    *,
    abstention_policy_version: str,
    baseline_run_id: str,
    baseline_experiment_sha256: str,
    model_semantic_sha256: str,
    predictions_validation_sha256: str,
    predictions_test_sha256: str,
    threshold_policy_id: str,
    threshold_policy_sha256: str,
    classification_threshold_basis_points: int,
    confidence_definition: str,
    abstention_grid: ThresholdGridConfig,
    minimum_coverage: float,
    selection_rule_version: str,
    metric_contract_version: str,
) -> str:
    """Hash the canonical policy-input payload that fully determines the generated policy."""
    payload = {
        "abstention_policy_version": abstention_policy_version,
        "abstention_grid": identity_abstention_grid_payload(abstention_grid),
        "baseline_experiment_sha256": baseline_experiment_sha256,
        "baseline_run_id": baseline_run_id,
        "classification_threshold_basis_points": classification_threshold_basis_points,
        "confidence_definition": confidence_definition,
        "metric_contract_version": metric_contract_version,
        "minimum_coverage": minimum_coverage,
        "model_semantic_sha256": model_semantic_sha256,
        "predictions_test_sha256": predictions_test_sha256,
        "predictions_validation_sha256": predictions_validation_sha256,
        "selection_rule_version": selection_rule_version,
        "threshold_policy_id": threshold_policy_id,
        "threshold_policy_sha256": threshold_policy_sha256,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_policy_id(threshold_policy_id: str, policy_input_sha256: str) -> str:
    """Derive the deterministic policy id from the policy-input hash."""
    return f"{threshold_policy_id}-ap{ABSTENTION_POLICY_VERSION}-{policy_input_sha256[:12]}"


def handled_metrics_from_split_metrics(metrics: SplitMetrics) -> HandledMetrics:
    aggregate = metrics.aggregate
    return HandledMetrics(
        subset_accuracy=aggregate.subset_accuracy,
        samples_f1=aggregate.samples_f1,
        micro_precision=aggregate.micro_precision,
        micro_recall=aggregate.micro_recall,
        micro_f1=aggregate.micro_f1,
        macro_precision=aggregate.macro_precision,
        macro_recall=aggregate.macro_recall,
        macro_f1=aggregate.macro_f1,
        mean_predicted_label_cardinality=aggregate.mean_predicted_label_cardinality,
        mean_true_label_cardinality=aggregate.mean_true_label_cardinality,
        false_positive_count=sum(item.fp for item in metrics.per_label),
        false_negative_count=sum(item.fn for item in metrics.per_label),
    )


def empty_handled_metrics() -> HandledMetrics:
    return HandledMetrics()


def comparison_split_metrics_from_handled(
    *,
    handled_metrics: HandledMetrics,
    coverage: float,
    handled_count: int,
    abstention_basis_points: int | None = None,
    abstention_threshold: float | None = None,
) -> ComparisonSplitMetrics:
    return ComparisonSplitMetrics(
        abstention_basis_points=abstention_basis_points,
        abstention_threshold=abstention_threshold,
        coverage=coverage,
        handled_count=handled_count,
        subset_accuracy=handled_metrics.subset_accuracy,
        samples_f1=handled_metrics.samples_f1,
        micro_f1=handled_metrics.micro_f1,
        macro_f1=handled_metrics.macro_f1,
        mean_predicted_label_cardinality=handled_metrics.mean_predicted_label_cardinality,
        mean_true_label_cardinality=handled_metrics.mean_true_label_cardinality,
        false_positive_count=handled_metrics.false_positive_count,
        false_negative_count=handled_metrics.false_negative_count,
    )
