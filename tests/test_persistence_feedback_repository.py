"""Tests for the SQLAlchemy feedback repository."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from repotriage.persistence.database import create_session_factory, init_database
from repotriage.persistence.errors import PersistenceError
from repotriage.persistence.feedback_repository import SqlAlchemyFeedbackRepository
from repotriage.persistence.models import FeedbackEvent
from repotriage.persistence.schemas import FeedbackRequest, InferenceArtifactsInput
from tests.helpers import (
    TEST_ABSTENTION_POLICY_ID,
    TEST_BASELINE_RUN_ID,
    TEST_MODEL_DATASET_ID,
    TEST_RETRIEVAL_RUN_ID,
    TEST_THRESHOLD_POLICY_ID,
    sqlite_database_url,
)


def _repository(tmp_path) -> SqlAlchemyFeedbackRepository:
    engine = create_engine(sqlite_database_url(tmp_path), future=True)
    init_database(engine)
    return SqlAlchemyFeedbackRepository(
        session_factory=create_session_factory(engine),
        engine=engine,
    )


def _request() -> FeedbackRequest:
    return FeedbackRequest(
        repository="pandas-dev/pandas",
        issue_number=12345,
        issue_title="BUG: example",
        issue_body_preview="Preview text",
        predicted_labels=["Indexing"],
        accepted_labels=["Bug", "Indexing"],
        rejected_labels=[],
        review_action="corrected",
        reviewer_note="Should also include Bug.",
        inference_artifacts=InferenceArtifactsInput(
            model_dataset_id=TEST_MODEL_DATASET_ID,
            baseline_run_id=TEST_BASELINE_RUN_ID,
            threshold_policy_id=TEST_THRESHOLD_POLICY_ID,
            abstention_policy_id=TEST_ABSTENTION_POLICY_ID,
            retrieval_run_id=TEST_RETRIEVAL_RUN_ID,
        ),
    )


def test_store_returns_feedback_id_and_created_at(tmp_path) -> None:
    repository = _repository(tmp_path)
    response = repository.store(_request())

    uuid.UUID(response.feedback_id)
    assert response.created_at.endswith("Z")
    assert response.status == "stored"


def test_store_persists_all_fields(tmp_path) -> None:
    repository = _repository(tmp_path)
    response = repository.store(_request())

    engine = create_engine(sqlite_database_url(tmp_path), future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_factory() as session:
        row = session.get(FeedbackEvent, response.feedback_id)
        assert row is not None
        assert row.repository == "pandas-dev/pandas"
        assert row.issue_number == 12345
        assert row.issue_title == "BUG: example"
        assert row.issue_body_preview == "Preview text"
        assert row.predicted_labels == ["Indexing"]
        assert row.accepted_labels == ["Bug", "Indexing"]
        assert row.rejected_labels == []
        assert row.review_action == "corrected"
        assert row.reviewer_note == "Should also include Bug."
        assert row.inference_artifacts == {
            "model_dataset_id": TEST_MODEL_DATASET_ID,
            "baseline_run_id": TEST_BASELINE_RUN_ID,
            "threshold_policy_id": TEST_THRESHOLD_POLICY_ID,
            "abstention_policy_id": TEST_ABSTENTION_POLICY_ID,
            "retrieval_run_id": TEST_RETRIEVAL_RUN_ID,
        }


def test_store_allows_multiple_events_for_same_issue(tmp_path) -> None:
    repository = _repository(tmp_path)
    first = repository.store(_request())
    second = repository.store(_request())

    assert first.feedback_id != second.feedback_id

    engine = create_engine(sqlite_database_url(tmp_path), future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_factory() as session:
        rows = session.scalars(
            select(FeedbackEvent).where(
                FeedbackEvent.repository == "pandas-dev/pandas",
                FeedbackEvent.issue_number == 12345,
            )
        ).all()
        assert len(rows) == 2


def test_store_sqlalchemy_error_raises_persistence_error(tmp_path, monkeypatch) -> None:
    repository = _repository(tmp_path)
    failing_session = MagicMock()
    failing_session.__enter__ = MagicMock(return_value=failing_session)
    failing_session.__exit__ = MagicMock(return_value=False)
    failing_session.commit.side_effect = SQLAlchemyError("database write failed")

    monkeypatch.setattr(
        repository,
        "_session_factory",
        lambda: failing_session,
    )

    with pytest.raises(PersistenceError, match="Failed to store feedback."):
        repository.store(_request())
