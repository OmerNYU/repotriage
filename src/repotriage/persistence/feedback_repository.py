"""Repository abstraction for storing maintainer feedback events."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from repotriage.dataset.models import format_utc_datetime
from repotriage.persistence.errors import PersistenceError
from repotriage.persistence.models import FEEDBACK_EVENT_SCHEMA_VERSION, FeedbackEvent
from repotriage.persistence.schemas import FeedbackRequest, FeedbackResponse


class FeedbackRepository(Protocol):
    """Store maintainer feedback events."""

    def store(self, body: FeedbackRequest) -> FeedbackResponse:
        """Persist one feedback event and return its acknowledgement."""


class SqlAlchemyFeedbackRepository:
    """SQLAlchemy-backed feedback repository."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        engine: Engine | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine

    def dispose(self) -> None:
        """Release database engine resources."""
        if self._engine is not None:
            self._engine.dispose()

    def store(self, body: FeedbackRequest) -> FeedbackResponse:
        """Persist one feedback event and return its acknowledgement."""
        feedback_id = str(uuid.uuid4())
        created_at = datetime.now(UTC)

        event = FeedbackEvent(
            id=feedback_id,
            created_at=created_at,
            schema_version=FEEDBACK_EVENT_SCHEMA_VERSION,
            repository=body.repository,
            issue_number=body.issue_number,
            issue_title=body.issue_title,
            issue_body_preview=body.issue_body_preview,
            predicted_labels=list(body.predicted_labels),
            accepted_labels=list(body.accepted_labels),
            rejected_labels=list(body.rejected_labels),
            review_action=body.review_action,
            reviewer_note=body.reviewer_note,
            inference_artifacts=body.inference_artifacts.model_dump(),
        )

        try:
            with self._session_factory() as session:
                session.add(event)
                session.commit()
        except SQLAlchemyError as exc:
            raise PersistenceError("Failed to store feedback.") from exc

        return FeedbackResponse(
            feedback_id=feedback_id,
            created_at=format_utc_datetime(created_at),
        )
