"""Database engine and session factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from repotriage.persistence.errors import DatabaseUnavailableError
from repotriage.persistence.models import Base

if TYPE_CHECKING:
    from repotriage.persistence.feedback_repository import SqlAlchemyFeedbackRepository


def create_database_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for the given database URL."""
    try:
        return create_engine(database_url, future=True)
    except ModuleNotFoundError as exc:
        if "psycopg" in str(exc) or database_url.startswith("postgresql"):
            raise DatabaseUnavailableError(
                "PostgreSQL driver not installed. Install with: pip install -e '.[db]'"
            ) from exc
        raise DatabaseUnavailableError(f"Unable to create database engine: {exc}") from exc
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"Unable to create database engine: {exc}") from exc


def check_database_connection(engine: Engine) -> None:
    """Verify that the database is reachable."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"Database connection failed: {exc}") from exc


def init_database(engine: Engine) -> None:
    """Create persistence tables when they do not already exist."""
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        raise DatabaseUnavailableError(f"Database initialization failed: {exc}") from exc


def create_session_factory(engine: Engine) -> Callable[[], Session]:
    """Build a session factory bound to the given engine."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def create_feedback_repository(database_url: str) -> SqlAlchemyFeedbackRepository:
    """Initialize persistence and return a feedback repository."""
    from repotriage.persistence.feedback_repository import SqlAlchemyFeedbackRepository

    engine = create_database_engine(database_url)
    check_database_connection(engine)
    init_database(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyFeedbackRepository(session_factory=session_factory, engine=engine)
