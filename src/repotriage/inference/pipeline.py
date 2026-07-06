"""End-to-end local issue inference orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from repotriage.github.models import RepositoryRef
from repotriage.inference.abstention import decide_abstention
from repotriage.inference.artifact_loader import LoadedInferenceBundle, load_inference_bundle
from repotriage.inference.classifier import score_issue
from repotriage.inference.models import (
    ArtifactReferences,
    ClassificationResult,
    InferenceInputSummary,
    InferenceIssueInput,
    InferenceResponse,
    InferenceWarning,
    ReproducibilityMetadata,
    RetrievalResult,
)
from repotriage.inference.retrieval import search_query_text
from repotriage.inference.text import (
    body_preview,
    build_inference_feature_text,
    feature_text_sha256,
)
from repotriage.inference.thresholding import apply_classification_threshold
from repotriage.inference.validators import validate_inference_response


def _collect_warnings(
    issue_input: InferenceIssueInput,
    *,
    predicted_label_count: int,
) -> list[InferenceWarning]:
    warnings: list[InferenceWarning] = []
    if issue_input.title == "":
        warnings.append("empty_title")
    if issue_input.body == "":
        warnings.append("empty_body")
    if predicted_label_count == 0:
        warnings.append("no_labels_predicted")
    return warnings


def infer_issue(
    bundle: LoadedInferenceBundle,
    issue_input: InferenceIssueInput,
) -> InferenceResponse:
    """Run the full inference pathway for one issue-like input."""
    top_k = issue_input.top_k if issue_input.top_k is not None else bundle.config.default_top_k

    feature_text = build_inference_feature_text(
        issue_input.title,
        issue_input.body,
        text_representation_version=bundle.config.text_representation_version,
    )

    y_score = score_issue(bundle.model, feature_text)
    scores, predicted_labels, y_pred = apply_classification_threshold(
        labels=bundle.label_order,
        y_score=y_score,
        threshold=bundle.classification_threshold,
        threshold_basis_points=bundle.classification_threshold_basis_points,
    )

    abstention = decide_abstention(
        y_score=y_score,
        y_pred=y_pred,
        classification_threshold=bundle.classification_threshold,
        abstention_threshold=bundle.abstention_threshold,
        abstention_threshold_basis_points=bundle.abstention_threshold_basis_points,
        confidence_definition=bundle.confidence_definition,
    )

    predicted_label_names = [item.label for item in predicted_labels]
    similar_issues = search_query_text(
        bundle.retrieval_index,
        feature_text,
        top_k=top_k,
        predicted_labels=predicted_label_names,
    )

    warnings = _collect_warnings(issue_input, predicted_label_count=len(predicted_labels))

    response = InferenceResponse(
        repository=bundle.repository.full_name,
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        input=InferenceInputSummary(
            title=issue_input.title,
            body_preview=body_preview(issue_input.body),
            feature_text_sha256=feature_text_sha256(feature_text),
            text_representation_version=bundle.config.text_representation_version,
        ),
        classification=ClassificationResult(
            label_order=list(bundle.label_order),
            scores=scores,
            threshold=bundle.classification_threshold,
            threshold_basis_points=bundle.classification_threshold_basis_points,
            predicted_labels=predicted_labels,
        ),
        abstention=abstention,
        retrieval=RetrievalResult(
            top_k=top_k,
            similar_issues=similar_issues,
        ),
        artifacts=ArtifactReferences(
            model_dataset_id=bundle.config.model_dataset_id,
            baseline_run_id=bundle.config.baseline_run_id,
            threshold_policy_id=bundle.config.threshold_policy_id,
            abstention_policy_id=bundle.config.abstention_policy_id,
            retrieval_run_id=bundle.config.retrieval_run_id,
        ),
        reproducibility=ReproducibilityMetadata(
            inference_config_path=str(bundle.config_path),
            model_semantic_sha256=bundle.baseline_manifest.model_semantic_sha256,
            index_semantic_sha256=bundle.retrieval_manifest.index_semantic_sha256,
            baseline_experiment_sha256=bundle.baseline_manifest.baseline_experiment_sha256,
            numerical_environment_sha256=bundle.baseline_manifest.numerical_environment_sha256,
            serialization_security_warning=bundle.serialization_security_warning,
        ),
        warnings=warnings,
    )
    validate_inference_response(response)
    return response


def infer_issue_from_config(
    config_path: Path,
    repository: RepositoryRef,
    issue_input: InferenceIssueInput,
    **loader_kwargs,
) -> InferenceResponse:
    """Load the inference bundle from config and score one issue."""
    bundle = load_inference_bundle(config_path, repository=repository, **loader_kwargs)
    return infer_issue(bundle, issue_input)
