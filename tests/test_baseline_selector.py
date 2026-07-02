"""Tests for validation candidate selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from repotriage.baseline.config import load_baseline_config
from repotriage.baseline.models import (
    AggregateMetrics,
    BaselineCandidateConfig,
    LogRegParams,
    PerLabelMetric,
    SplitMetrics,
    TfidfParams,
)
from repotriage.baseline.selector import CandidateScore, select_candidate
from tests.helpers import write_baseline_config


def _metrics(macro_ap: float, macro_f1: float, micro_f1: float) -> SplitMetrics:
    return SplitMetrics(
        split="validation",
        threshold=0.5,
        score_type="probability_estimates",
        per_label=[
            PerLabelMetric(
                label="Bug",
                support=1,
                prevalence=0.5,
                predicted_positives=1,
                tp=1,
                fp=0,
                fn=0,
                tn=1,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                average_precision=macro_ap,
            )
        ],
        aggregate=AggregateMetrics(
            macro_average_precision=macro_ap,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            macro_label_count=1,
            macro_precision_denominator=1,
            macro_recall_denominator=1,
            macro_f1_denominator=1,
            undefined_precision_label_count=0,
            undefined_recall_label_count=0,
            undefined_f1_label_count=0,
            macro_average_precision_label_count=1,
            subset_accuracy=0.5,
            hamming_loss=0.25,
            mean_true_label_cardinality=0.5,
            mean_predicted_label_cardinality=0.5,
            fraction_no_prediction=0.5,
            fraction_any_prediction=0.5,
            record_count=2,
        ),
    )


def _score_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[1]], dtype=np.int8),
        np.array([[0.9]], dtype=np.float64),
    )


def _candidate(candidate_id: str, ngram_high: int, balanced: bool) -> BaselineCandidateConfig:
    return BaselineCandidateConfig(
        candidate_id=candidate_id,
        tfidf=TfidfParams(ngram_range=(1, ngram_high), min_df=1),
        logreg=LogRegParams(
            C=1.0,
            max_iter=100,
            class_weight="balanced" if balanced else None,
        ),
    )


def test_select_highest_macro_ap(tmp_path: Path) -> None:
    config, _, _, _ = load_baseline_config(write_baseline_config(tmp_path / "baseline.json"))
    preds, scores = _score_arrays()
    scored = [
        CandidateScore(
            candidate_id="c1_unigram",
            candidate=_candidate("c1_unigram", 1, False),
            metrics=_metrics(0.4, 0.4, 0.4),
            vocabulary_size=10,
            convergence_warnings=[],
            label_convergence={},
            validation_predictions=preds,
            validation_scores=scores,
        ),
        CandidateScore(
            candidate_id="c2_bigram",
            candidate=_candidate("c2_bigram", 2, False),
            metrics=_metrics(0.6, 0.3, 0.3),
            vocabulary_size=12,
            convergence_warnings=[],
            label_convergence={},
            validation_predictions=preds,
            validation_scores=scores,
        ),
    ]
    winner_id, audit = select_candidate(config=config, scored_candidates=scored)
    assert winner_id == "c2_bigram"
    assert audit.winner_candidate_id == "c2_bigram"


def test_tie_break_prefers_lower_complexity(tmp_path: Path) -> None:
    config, _, _, _ = load_baseline_config(write_baseline_config(tmp_path / "baseline.json"))
    preds, scores = _score_arrays()
    scored = [
        CandidateScore(
            candidate_id="c2_bigram",
            candidate=_candidate("c2_bigram", 2, False),
            metrics=_metrics(0.5, 0.5, 0.5),
            vocabulary_size=12,
            convergence_warnings=[],
            label_convergence={},
            validation_predictions=preds,
            validation_scores=scores,
        ),
        CandidateScore(
            candidate_id="c1_unigram",
            candidate=_candidate("c1_unigram", 1, False),
            metrics=_metrics(0.5, 0.5, 0.5),
            vocabulary_size=10,
            convergence_warnings=[],
            label_convergence={},
            validation_predictions=preds,
            validation_scores=scores,
        ),
    ]
    winner_id, _audit = select_candidate(config=config, scored_candidates=scored)
    assert winner_id == "c1_unigram"
