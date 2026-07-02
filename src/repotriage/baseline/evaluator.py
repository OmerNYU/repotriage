"""Multilabel metric contract and prediction evaluation."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    hamming_loss,
    roc_auc_score,
)

from repotriage.baseline.models import (
    METRIC_CONTRACT_VERSION,
    SAMPLES_F1_EMPTY_EMPTY_POLICY,
    AggregateMetrics,
    BaselineCorruptionError,
    PerLabelMetric,
    PredictionRecord,
    ScoreType,
    SplitMetrics,
)
from repotriage.baseline.models_ml import AllZeroPredictor, TfidfMultiLabelLogRegModel
from repotriage.baseline.scores import validate_score_matrix
from repotriage.model_dataset.models import ModelReadyRecord, SplitName


def _vector_to_labels(vector: list[int] | np.ndarray, labels: list[str]) -> list[str]:
    return [label for label, value in zip(labels, vector, strict=True) if int(value) == 1]


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _samples_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: list[float] = []
    for row_index in range(y_true.shape[0]):
        true_row = y_true[row_index]
        pred_row = y_pred[row_index]
        true_positive = int(np.logical_and(true_row == 1, pred_row == 1).sum())
        true_count = int(true_row.sum())
        pred_count = int(pred_row.sum())
        if true_count == 0 and pred_count == 0:
            scores.append(1.0)
            continue
        if true_count == 0 or pred_count == 0:
            scores.append(0.0)
            continue
        false_positive = int(np.logical_and(true_row == 0, pred_row == 1).sum())
        false_negative = int(np.logical_and(true_row == 1, pred_row == 0).sum())
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision + recall > 0:
            scores.append(2 * precision * recall / (precision + recall))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def _per_label_metrics(
    *,
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    record_count: int,
) -> PerLabelMetric:
    support = int(y_true.sum())
    predicted_positives = int(y_pred.sum())
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    if precision is None:
        precision_undefined_reason = "no_positive_predictions"
    else:
        precision_undefined_reason = None
    if recall is None:
        recall_undefined_reason = "no_positive_support"
    else:
        recall_undefined_reason = None

    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
    else:
        f1 = None

    average_precision: float | None = None
    if y_score is not None:
        if support > 0:
            average_precision = float(average_precision_score(y_true, y_score))

    roc_auc: float | None = None
    roc_auc_undefined_reason: str | None = None
    if y_score is not None:
        unique = np.unique(y_true)
        if unique.size < 2:
            roc_auc_undefined_reason = "single_class"
        else:
            roc_auc = float(roc_auc_score(y_true, y_score))

    return PerLabelMetric(
        label=label,
        support=support,
        prevalence=support / record_count if record_count else 0.0,
        predicted_positives=predicted_positives,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        precision_undefined_reason=precision_undefined_reason,
        recall_undefined_reason=recall_undefined_reason,
        average_precision=average_precision,
        roc_auc=roc_auc,
        roc_auc_undefined_reason=roc_auc_undefined_reason,
    )


def _macro_zero_filled(values: list[tuple[str, float | None]]) -> tuple[float, int, int, list[str]]:
    label_count = len(values)
    skipped = [label for label, value in values if value is None]
    filled = [0.0 if value is None else float(value) for _, value in values]
    if not filled:
        return 0.0, 0, 0, skipped
    return float(np.mean(filled)), label_count, len(skipped), skipped


def _macro_defined_only(
    values: list[tuple[str, float | None]],
) -> tuple[float | None, int, list[str]]:
    defined = [(label, value) for label, value in values if value is not None]
    skipped = [label for label, value in values if value is None]
    if not defined:
        return None, 0, skipped
    return float(np.mean([value for _, value in defined])), len(defined), skipped


def compute_split_metrics(
    *,
    split: SplitName,
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    threshold: float,
    score_type: ScoreType,
) -> SplitMetrics:
    record_count = y_true.shape[0]
    per_label = [
        _per_label_metrics(
            label=label,
            y_true=y_true[:, index],
            y_pred=y_pred[:, index],
            y_score=None if y_score is None else y_score[:, index],
            record_count=record_count,
        )
        for index, label in enumerate(labels)
    ]

    micro_precision = _safe_divide(
        sum(item.tp for item in per_label),
        sum(item.tp + item.fp for item in per_label),
    )
    micro_recall = _safe_divide(
        sum(item.tp for item in per_label),
        sum(item.tp + item.fn for item in per_label),
    )
    if (
        micro_precision is not None
        and micro_recall is not None
        and (micro_precision + micro_recall) > 0
    ):
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = None

    macro_precision, macro_label_count, undefined_precision, skipped_precision = _macro_zero_filled(
        [(item.label, item.precision) for item in per_label]
    )
    macro_recall, _, undefined_recall, skipped_recall = _macro_zero_filled(
        [(item.label, item.recall) for item in per_label]
    )
    macro_f1, _, undefined_f1, skipped_f1 = _macro_zero_filled(
        [(item.label, item.f1) for item in per_label]
    )
    macro_precision_defined_only, macro_precision_denominator, _ = _macro_defined_only(
        [(item.label, item.precision) for item in per_label]
    )
    macro_recall_defined_only, macro_recall_denominator, _ = _macro_defined_only(
        [(item.label, item.recall) for item in per_label]
    )
    macro_f1_defined_only, macro_f1_denominator, _ = _macro_defined_only(
        [(item.label, item.f1) for item in per_label]
    )

    supports = np.array([item.support for item in per_label], dtype=np.float64)
    f1_values = np.array(
        [item.f1 if item.f1 is not None else 0.0 for item in per_label], dtype=np.float64
    )
    weighted_f1 = float(np.average(f1_values, weights=supports)) if supports.sum() > 0 else None

    samples_f1 = _samples_f1(y_true, y_pred)

    subset_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    hamming = float(hamming_loss(y_true, y_pred))

    ap_values = [
        (item.label, item.average_precision)
        for item in per_label
        if item.average_precision is not None
    ]
    macro_average_precision_label_count = len(ap_values)
    macro_average_precision = (
        float(np.mean([value for _, value in ap_values])) if ap_values else None
    )
    if y_score is not None:
        micro_average_precision = float(
            average_precision_score(y_true, y_score, average="micro")
        )
    else:
        micro_average_precision = None

    true_cardinality = y_true.sum(axis=1)
    pred_cardinality = y_pred.sum(axis=1)

    aggregate = AggregateMetrics(
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        macro_precision_defined_only=macro_precision_defined_only,
        macro_recall_defined_only=macro_recall_defined_only,
        macro_f1_defined_only=macro_f1_defined_only,
        macro_label_count=macro_label_count,
        macro_precision_denominator=macro_precision_denominator,
        macro_recall_denominator=macro_recall_denominator,
        macro_f1_denominator=macro_f1_denominator,
        undefined_precision_label_count=undefined_precision,
        undefined_recall_label_count=undefined_recall,
        undefined_f1_label_count=undefined_f1,
        macro_labels_skipped_precision=skipped_precision,
        macro_labels_skipped_recall=skipped_recall,
        macro_labels_skipped_f1=skipped_f1,
        weighted_f1=weighted_f1,
        samples_f1=samples_f1,
        samples_f1_empty_empty_policy=SAMPLES_F1_EMPTY_EMPTY_POLICY,
        subset_accuracy=subset_accuracy,
        hamming_loss=hamming,
        macro_average_precision=macro_average_precision,
        macro_average_precision_label_count=macro_average_precision_label_count,
        micro_average_precision=micro_average_precision,
        mean_true_label_cardinality=float(true_cardinality.mean()) if record_count else 0.0,
        mean_predicted_label_cardinality=float(pred_cardinality.mean()) if record_count else 0.0,
        fraction_no_prediction=float(np.mean(pred_cardinality == 0)) if record_count else 0.0,
        fraction_any_prediction=float(np.mean(pred_cardinality > 0)) if record_count else 0.0,
        record_count=record_count,
    )

    return SplitMetrics(
        metric_contract_version=METRIC_CONTRACT_VERSION,
        split=split,
        threshold=threshold,
        score_type=score_type,
        per_label=per_label,
        aggregate=aggregate,
    )


def evaluate_model_on_split(
    *,
    model: TfidfMultiLabelLogRegModel,
    records: list[ModelReadyRecord],
    texts: list[str],
    targets: np.ndarray,
    labels: list[str],
    split: SplitName,
) -> tuple[SplitMetrics, np.ndarray, np.ndarray]:
    scores = model.predict_proba_matrix(texts)
    validate_score_matrix(scores, target_count=len(labels))
    predictions = (scores >= model.threshold).astype(np.int8)
    metrics = compute_split_metrics(
        split=split,
        labels=labels,
        y_true=targets,
        y_pred=predictions,
        y_score=scores,
        threshold=model.threshold,
        score_type="probability_estimates",
    )
    return metrics, predictions, scores


def evaluate_frozen_candidate(
    *,
    model: TfidfMultiLabelLogRegModel,
    test_records: list[ModelReadyRecord],
    test_texts: list[str],
    test_targets: np.ndarray,
    labels: list[str],
) -> tuple[SplitMetrics, np.ndarray, np.ndarray]:
    """Evaluate the frozen winner on the held-out test split."""
    return evaluate_model_on_split(
        model=model,
        records=test_records,
        texts=test_texts,
        targets=test_targets,
        labels=labels,
        split="test",
    )


def evaluate_all_zero_on_split(
    *,
    predictor: AllZeroPredictor,
    records: list[ModelReadyRecord],
    targets: np.ndarray,
    labels: list[str],
    split: SplitName,
    threshold: float,
) -> SplitMetrics:
    predictions = predictor.predict_matrix(len(records))
    return compute_split_metrics(
        split=split,
        labels=labels,
        y_true=targets,
        y_pred=predictions,
        y_score=None,
        threshold=threshold,
        score_type="none",
    )


def build_prediction_records(
    *,
    repository: str,
    model_dataset_id: str,
    baseline_run_id: str,
    labels: list[str],
    records: list[ModelReadyRecord],
    predictions: np.ndarray,
    scores: np.ndarray | None,
    split: SplitName,
    threshold: float | None,
    score_type: ScoreType,
    candidate_id: str | None = None,
) -> list[PredictionRecord]:
    target_count = len(labels)
    output: list[PredictionRecord] = []
    for row_index, record in enumerate(records):
        true_vector = [int(value) for value in record.target_vector]
        predicted_vector = [int(value) for value in predictions[row_index].tolist()]
        score_vector = None
        if scores is not None:
            score_vector = [float(value) for value in scores[row_index].tolist()]
            validate_score_matrix(
                np.array([score_vector], dtype=np.float64),
                target_count=target_count,
            )
        if threshold is not None and score_vector is not None:
            for index, score in enumerate(score_vector):
                expected = int(score >= threshold)
                if predicted_vector[index] != expected:
                    raise ValueError(
                        f"predicted_vector[{index}]={predicted_vector[index]} does not match "
                        f"int(score>={threshold})={expected}"
                    )
        output.append(
            PredictionRecord(
                candidate_id=candidate_id,
                repository=repository,
                model_dataset_id=model_dataset_id,
                baseline_run_id=baseline_run_id,
                issue_id=record.issue_id,
                issue_number=record.issue_number,
                split=split,
                true_labels=_vector_to_labels(true_vector, labels),
                true_vector=true_vector,
                predicted_labels=_vector_to_labels(predicted_vector, labels),
                predicted_vector=predicted_vector,
                score_type=score_type,
                threshold=threshold,
                score_vector=score_vector,
            )
        )
    return output


def _validate_prediction_record_strict(record: PredictionRecord, labels: list[str]) -> None:
    if len(record.true_vector) != len(labels):
        raise BaselineCorruptionError("prediction true_vector has unexpected length")
    if len(record.predicted_vector) != len(labels):
        raise BaselineCorruptionError("prediction predicted_vector has unexpected length")
    for field_name in ("true_vector", "predicted_vector"):
        vector = getattr(record, field_name)
        for index, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, int):
                raise BaselineCorruptionError(
                    f"{field_name}[{index}] must be an int, got {type(value).__name__}"
                )
            if value not in (0, 1):
                raise BaselineCorruptionError(f"{field_name}[{index}] must be 0 or 1")
    if record.score_vector is not None:
        if len(record.score_vector) != len(labels):
            raise BaselineCorruptionError("prediction score_vector has unexpected length")
        for index, score in enumerate(record.score_vector):
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise BaselineCorruptionError(
                    f"score_vector[{index}] must be numeric, got {type(score).__name__}"
                )
            if not math.isfinite(float(score)):
                raise BaselineCorruptionError(f"score_vector[{index}] must be finite")
            if float(score) < 0.0 or float(score) > 1.0:
                raise BaselineCorruptionError(f"score_vector[{index}] must lie in [0, 1]")
        if record.threshold is not None:
            for index, score in enumerate(record.score_vector):
                expected = int(float(score) >= record.threshold)
                if record.predicted_vector[index] != expected:
                    raise BaselineCorruptionError(
                        "predicted_vector does not match thresholded scores"
                    )


def metrics_from_predictions(
    *,
    labels: list[str],
    records: list[PredictionRecord],
    split: SplitName,
) -> SplitMetrics:
    if not records:
        raise ValueError("records must not be empty")
    threshold = records[0].threshold
    score_type = records[0].score_type
    for record in records:
        _validate_prediction_record_strict(record, labels)
    y_true = np.array([record.true_vector for record in records], dtype=np.int8)
    y_pred = np.array([record.predicted_vector for record in records], dtype=np.int8)
    if score_type == "none":
        y_score = None
    else:
        y_score = np.array(
            [record.score_vector for record in records],
            dtype=np.float64,
        )
        validate_score_matrix(y_score, target_count=len(labels))
    return compute_split_metrics(
        split=split,
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        threshold=threshold if threshold is not None else 0.5,
        score_type=score_type,
    )
