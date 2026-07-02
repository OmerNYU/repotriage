"""Explicit per-label logistic regression multilabel models."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from repotriage.baseline.features import candidate_vectorizer, fit_vectorizer, transform_texts
from repotriage.baseline.models import (
    BASELINE_VERSION,
    MODEL_SEMANTIC_CONTRACT_VERSION,
    BaselineCandidateConfig,
    BaselineTrainingError,
)
from repotriage.baseline.runtime import numerical_thread_limits
from repotriage.baseline.scores import validate_score_matrix


def _json_safe(value: Any) -> Any:
    """Convert sklearn/numpy values into JSON-serializable structures."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _array_fingerprint(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def model_semantic_sha256(model: TfidfMultiLabelLogRegModel) -> str:
    """Canonical fingerprint over inference-relevant fitted model state."""
    vectorizer = model.vectorizer
    vocabulary = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[0])
    estimator_payloads: list[dict[str, Any]] = []
    for estimator in model.estimators:
        estimator_payloads.append(
            {
                "class_name": type(estimator).__name__,
                "classes_": _array_fingerprint(np.asarray(estimator.classes_)),
                "coef_": _array_fingerprint(estimator.coef_),
                "intercept_": _array_fingerprint(estimator.intercept_),
                "n_features_in_": int(estimator.n_features_in_),
                "n_iter_": [int(value) for value in estimator.n_iter_.tolist()],
                "params": _json_safe(estimator.get_params(deep=True)),
            }
        )

    payload = {
        "baseline_version": BASELINE_VERSION,
        "estimators": estimator_payloads,
        "labels": list(model.labels),
        "model_semantic_contract_version": MODEL_SEMANTIC_CONTRACT_VERSION,
        "score_type": "probability_estimates",
        "threshold": float(model.threshold),
        "vectorizer": {
            "class_name": type(vectorizer).__name__,
            "idf_": _array_fingerprint(vectorizer.idf_),
            "params": _json_safe(vectorizer.get_params()),
            "vocabulary": [[term, int(index)] for term, index in vocabulary],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class LabelTrainingReport:
    label: str
    n_iter: int
    converged: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrainingReport:
    vocabulary_size: int
    label_reports: list[LabelTrainingReport]
    convergence_warnings: list[str]


@dataclass
class TfidfMultiLabelLogRegModel:
    """TF-IDF vectorizer plus one logistic regression estimator per label."""

    labels: list[str]
    vectorizer: TfidfVectorizer
    estimators: list[LogisticRegression]
    threshold: float
    random_state: int

    def _positive_column_index(self, estimator: LogisticRegression) -> int:
        classes = set(int(value) for value in estimator.classes_)
        if classes != {0, 1}:
            raise ValueError(f"estimator.classes_ must be exactly {{0, 1}}; got {classes}")
        return list(estimator.classes_).index(1)

    def predict_proba_matrix(self, texts: list[str]) -> np.ndarray:
        n_samples = len(texts)
        n_labels = len(self.labels)
        scores = np.zeros((n_samples, n_labels), dtype=np.float64)
        with numerical_thread_limits():
            features = transform_texts(self.vectorizer, texts)
            for label_index, estimator in enumerate(self.estimators):
                proba = estimator.predict_proba(features)
                positive_index = self._positive_column_index(estimator)
                scores[:, label_index] = proba[:, positive_index]
        validate_score_matrix(scores, target_count=n_labels)
        return scores

    def predict_matrix(self, texts: list[str]) -> np.ndarray:
        scores = self.predict_proba_matrix(texts)
        return (scores >= self.threshold).astype(np.int8)

    def to_bundle(self) -> dict:
        return {
            "baseline_version": BASELINE_VERSION,
            "labels": self.labels,
            "vectorizer": self.vectorizer,
            "estimators": self.estimators,
            "threshold": self.threshold,
            "score_type": "probability_estimates",
        }


class AllZeroPredictor:
    """Dummy baseline that predicts no labels and emits no meaningful scores."""

    def __init__(self, *, labels: list[str]) -> None:
        self.labels = labels

    def predict_matrix(self, n_samples: int) -> np.ndarray:
        return np.zeros((n_samples, len(self.labels)), dtype=np.int8)


def _build_logreg(candidate: BaselineCandidateConfig, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=candidate.logreg.C,
        solver=candidate.logreg.solver,
        max_iter=candidate.logreg.max_iter,
        class_weight=candidate.logreg.class_weight,
        random_state=random_state,
    )


def train_model(
    *,
    candidate: BaselineCandidateConfig,
    labels: list[str],
    train_texts: list[str],
    train_targets: np.ndarray,
    random_state: int,
    threshold: float,
) -> tuple[TfidfMultiLabelLogRegModel, TrainingReport]:
    estimators: list[LogisticRegression] = []
    label_reports: list[LabelTrainingReport] = []
    convergence_warnings: list[str] = []

    with numerical_thread_limits():
        vectorizer = candidate_vectorizer(candidate)
        fit_vectorizer(vectorizer, train_texts)
        features = transform_texts(vectorizer, train_texts)

        for label_index, label in enumerate(labels):
            y = train_targets[:, label_index]
            unique = np.unique(y)
            if unique.size < 2:
                raise BaselineTrainingError(
                    f"Label {label!r} has only one class in training data; "
                    "binary logistic regression requires both classes."
                )
            estimator = _build_logreg(candidate, random_state)
            label_warnings: list[str] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(features, y)
                for item in caught:
                    if issubclass(item.category, UserWarning):
                        message = str(item.message)
                        label_warnings.append(message)
                        convergence_warnings.append(f"{label}: {message}")
            converged = estimator.n_iter_[0] < candidate.logreg.max_iter
            label_reports.append(
                LabelTrainingReport(
                    label=label,
                    n_iter=int(estimator.n_iter_[0]),
                    converged=converged,
                    warnings=label_warnings,
                )
            )
            estimators.append(estimator)

    model = TfidfMultiLabelLogRegModel(
        labels=labels,
        vectorizer=vectorizer,
        estimators=estimators,
        threshold=threshold,
        random_state=random_state,
    )
    report = TrainingReport(
        vocabulary_size=len(vectorizer.vocabulary_),
        label_reports=label_reports,
        convergence_warnings=convergence_warnings,
    )
    return model, report


def load_model_from_bundle(bundle: dict) -> TfidfMultiLabelLogRegModel:
    return TfidfMultiLabelLogRegModel(
        labels=list(bundle["labels"]),
        vectorizer=bundle["vectorizer"],
        estimators=list(bundle["estimators"]),
        threshold=float(bundle["threshold"]),
        random_state=0,
    )
