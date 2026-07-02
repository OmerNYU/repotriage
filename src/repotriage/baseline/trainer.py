"""Stage 1 baseline training for all predeclared candidates."""

from __future__ import annotations

from dataclasses import dataclass

from repotriage.baseline.config import BaselineConfigDocument
from repotriage.baseline.models import BaselineCandidateConfig
from repotriage.baseline.models_ml import TfidfMultiLabelLogRegModel, TrainingReport, train_model
from repotriage.baseline.reader import ModelReadySplits, TrainingSplits


@dataclass(frozen=True)
class TrainedCandidate:
    candidate: BaselineCandidateConfig
    model: TfidfMultiLabelLogRegModel
    training_report: TrainingReport


def train_candidate(
    *,
    candidate: BaselineCandidateConfig,
    splits: TrainingSplits | ModelReadySplits,
    random_state: int,
    threshold: float,
) -> TrainedCandidate:
    model, training_report = train_model(
        candidate=candidate,
        labels=splits.label_map.labels,
        train_texts=splits.train.texts,
        train_targets=splits.train.targets,
        random_state=random_state,
        threshold=threshold,
    )
    return TrainedCandidate(
        candidate=candidate,
        model=model,
        training_report=training_report,
    )


def train_all_candidates(
    *,
    config: BaselineConfigDocument,
    splits: TrainingSplits | ModelReadySplits,
) -> list[TrainedCandidate]:
    threshold = config.threshold_policy.threshold
    random_state = config.random_state
    return [
        train_candidate(
            candidate=candidate,
            splits=splits,
            random_state=random_state,
            threshold=threshold,
        )
        for candidate in config.candidates
    ]
