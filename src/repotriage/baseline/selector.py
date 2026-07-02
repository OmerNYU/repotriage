"""Validation-only candidate selection with deterministic tie-breaks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repotriage.baseline.config import BaselineConfigDocument
from repotriage.baseline.evaluator import (
    build_prediction_records,
    evaluate_all_zero_on_split,
    evaluate_model_on_split,
)
from repotriage.baseline.models import (
    BaselineCandidateConfig,
    CandidateValidationResult,
    PredictionRecord,
    SelectionAudit,
    SplitMetrics,
)
from repotriage.baseline.models_ml import AllZeroPredictor
from repotriage.baseline.reader import TrainingSplits
from repotriage.baseline.trainer import TrainedCandidate, train_all_candidates


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    candidate: BaselineCandidateConfig
    metrics: SplitMetrics
    vocabulary_size: int
    convergence_warnings: list[str]
    label_convergence: dict[str, int]
    validation_predictions: np.ndarray
    validation_scores: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    winner_id: str
    selection_audit: SelectionAudit
    scored_candidates: list[CandidateScore]
    dummy_metrics: SplitMetrics
    trained_winner: TrainedCandidate
    validation_prediction_records: list[PredictionRecord]


def _complexity_key(candidate: BaselineCandidateConfig) -> tuple[int, int, str]:
    ngram_high = candidate.tfidf.ngram_range[1]
    class_weight_rank = 0 if candidate.logreg.class_weight is None else 1
    return (ngram_high, class_weight_rank, candidate.candidate_id)


def _metric_value(metrics: SplitMetrics, name: str) -> float:
    aggregate = metrics.aggregate
    value = getattr(aggregate, name)
    if value is None:
        return float("-inf")
    return float(value)


def select_candidate(
    *,
    config: BaselineConfigDocument,
    scored_candidates: list[CandidateScore],
) -> tuple[str, SelectionAudit]:
    selectable = [item for item in scored_candidates if item.candidate_id != "dummy_all_zero"]
    if not selectable:
        raise ValueError("No selectable candidates provided")

    ranked = sorted(
        selectable,
        key=lambda item: (
            _metric_value(item.metrics, "macro_average_precision"),
            _metric_value(item.metrics, "macro_f1"),
            _metric_value(item.metrics, "micro_f1"),
            -_complexity_key(item.candidate)[0],
            -_complexity_key(item.candidate)[1],
            item.candidate_id,
        ),
        reverse=True,
    )

    winner = ranked[0]
    tie_break_steps: list[str] = []
    if len(ranked) > 1:
        runner_up = ranked[1]
        if _metric_value(winner.metrics, "macro_average_precision") == _metric_value(
            runner_up.metrics, "macro_average_precision"
        ):
            tie_break_steps.append("macro_average_precision_tied")
        if _metric_value(winner.metrics, "macro_f1") == _metric_value(
            runner_up.metrics, "macro_f1"
        ):
            tie_break_steps.append("macro_f1_tied")
        if _metric_value(winner.metrics, "micro_f1") == _metric_value(
            runner_up.metrics, "micro_f1"
        ):
            tie_break_steps.append("micro_f1_tied")

    audit = SelectionAudit(
        winner_candidate_id=winner.candidate_id,
        tie_break_steps=tie_break_steps,
        ranked_candidate_ids=[item.candidate_id for item in ranked],
    )
    return winner.candidate_id, audit


def to_candidate_validation_result(score: CandidateScore) -> CandidateValidationResult:
    return CandidateValidationResult(
        candidate_id=score.candidate_id,
        selectable=score.candidate_id != "dummy_all_zero",
        metrics=score.metrics,
        vocabulary_size=score.vocabulary_size,
        convergence_warnings=score.convergence_warnings,
        label_convergence=score.label_convergence,
    )


def run_candidate_selection(
    *,
    config: BaselineConfigDocument,
    splits: TrainingSplits,
    repository: str,
    model_dataset_id: str,
    baseline_run_id: str,
) -> SelectionResult:
    """Train all candidates, score on validation only, and freeze the winner."""
    labels = splits.label_map.labels
    threshold = config.threshold_policy.threshold
    trained_candidates = train_all_candidates(config=config, splits=splits)

    dummy = AllZeroPredictor(labels=labels)
    dummy_metrics = evaluate_all_zero_on_split(
        predictor=dummy,
        records=splits.validation.records,
        targets=splits.validation.targets,
        labels=labels,
        split="validation",
        threshold=threshold,
    )

    scored: list[CandidateScore] = []
    all_prediction_records: list[PredictionRecord] = []
    for trained in trained_candidates:
        metrics, predictions, scores = evaluate_model_on_split(
            model=trained.model,
            records=splits.validation.records,
            texts=splits.validation.texts,
            targets=splits.validation.targets,
            labels=labels,
            split="validation",
        )
        scored.append(
            CandidateScore(
                candidate_id=trained.candidate.candidate_id,
                candidate=trained.candidate,
                metrics=metrics,
                vocabulary_size=trained.training_report.vocabulary_size,
                convergence_warnings=trained.training_report.convergence_warnings,
                label_convergence={
                    report.label: report.n_iter
                    for report in trained.training_report.label_reports
                },
                validation_predictions=predictions,
                validation_scores=scores,
            )
        )
        all_prediction_records.extend(
            build_prediction_records(
                repository=repository,
                model_dataset_id=model_dataset_id,
                baseline_run_id=baseline_run_id,
                labels=labels,
                records=splits.validation.records,
                predictions=predictions,
                scores=scores,
                split="validation",
                threshold=threshold,
                score_type="probability_estimates",
                candidate_id=trained.candidate.candidate_id,
            )
        )

    winner_id, selection_audit = select_candidate(config=config, scored_candidates=scored)
    trained_winner = next(
        item for item in trained_candidates if item.candidate.candidate_id == winner_id
    )

    def _prediction_sort_key(record: PredictionRecord) -> tuple[int, str, int]:
        candidate_order = {
            candidate.candidate_id: index
            for index, candidate in enumerate(config.candidates)
        }
        candidate_index = candidate_order.get(record.candidate_id or "", 10_000)
        created_at = splits.validation.records[
            next(
                index
                for index, item in enumerate(splits.validation.records)
                if item.issue_id == record.issue_id
            )
        ].model_dump(mode="json")["created_at"]
        return (candidate_index, created_at, record.issue_id)

    all_prediction_records.sort(key=_prediction_sort_key)

    return SelectionResult(
        winner_id=winner_id,
        selection_audit=selection_audit,
        scored_candidates=scored,
        dummy_metrics=dummy_metrics,
        trained_winner=trained_winner,
        validation_prediction_records=all_prediction_records,
    )
