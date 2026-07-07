"""Integration tests for POST /api/v1/feedback with a real SQLite database."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from repotriage.api.app import create_app
from repotriage.api.settings import ApiSettings
from repotriage.persistence.models import FeedbackEvent
from tests.helpers import make_feedback_request_payload, make_test_bundle, sqlite_database_url


def test_feedback_endpoint_persists_row(tmp_path: Path) -> None:
    settings = ApiSettings(
        inference_config_path=Path("configs/test.json"),
        database_url=sqlite_database_url(tmp_path),
    )
    app = create_app(settings=settings, bundle=make_test_bundle())

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=make_feedback_request_payload())

    assert response.status_code == 201
    payload = response.json()
    feedback_id = payload["feedback_id"]

    engine = create_engine(sqlite_database_url(tmp_path), future=True)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with session_factory() as session:
        row = session.get(FeedbackEvent, feedback_id)
        assert row is not None
        assert row.repository == "pandas-dev/pandas"
        assert row.issue_number == 12345
        assert row.review_action == "corrected"
        assert row.predicted_labels == ["Indexing"]
        assert row.accepted_labels == ["Bug", "Indexing"]

        rows = session.scalars(select(FeedbackEvent)).all()
        assert len(rows) == 1
