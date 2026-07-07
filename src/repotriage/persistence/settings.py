"""Persistence settings and environment variable resolution."""

from __future__ import annotations

import os
from typing import Any

DATABASE_URL_ENV_VAR = "DATABASE_URL"
DEFAULT_DATABASE_URL = "sqlite:///./data/repotriage_feedback.db"


def resolve_database_url(*, cli_value: str | None = None) -> str:
    """Resolve the database URL with precedence: CLI flag > env var > default."""
    if cli_value is not None:
        return cli_value
    return os.environ.get(DATABASE_URL_ENV_VAR, DEFAULT_DATABASE_URL)


def database_url_from_namespace(args: Any) -> str:
    """Read an optional --database-url value from an argparse namespace."""
    return resolve_database_url(cli_value=getattr(args, "database_url", None))
