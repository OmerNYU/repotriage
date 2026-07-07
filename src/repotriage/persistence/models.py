"""SQLAlchemy ORM models for feedback persistence."""

from __future__ import annotations

from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

FEEDBACK_EVENT_SCHEMA_VERSION = "1"


class Base(DeclarativeBase):
    """Declarative base for persistence models."""


class FeedbackEvent(Base):
    """One maintainer review event tied to an inference prediction."""

    __tablename__ = "feedback_events"
    __table_args__ = (
        Index("ix_feedback_events_repository_issue_number", "repository", "issue_number"),
        Index("ix_feedback_events_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    schema_version: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        default=FEEDBACK_EVENT_SCHEMA_VERSION,
    )
    repository: Mapped[str] = mapped_column(String(256), nullable=False)
    issue_number: Mapped[int] = mapped_column(Integer, nullable=False)
    issue_title: Mapped[str] = mapped_column(Text, nullable=False)
    issue_body_preview: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    predicted_labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    accepted_labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    rejected_labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    review_action: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    inference_artifacts: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
