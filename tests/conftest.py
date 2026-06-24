"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from repotriage.github.models import RepositoryRef


@pytest.fixture
def repository() -> RepositoryRef:
    return RepositoryRef(owner="pandas-dev", name="pandas")


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"
