"""Generic atomic file I/O and best-effort cleanup helpers.

These utilities are domain-agnostic and shared across subsystems. Domain-specific
directory publication (raw-cache replacement, immutable snapshot publication) lives
in the subsystems that own those concerns.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def best_effort_remove(path: Path) -> None:
    """Remove a file, logging but swallowing OS errors."""
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        logger.warning("Failed to remove temporary file %s", path, exc_info=True)


def best_effort_remove_tree(path: Path) -> None:
    """Remove a directory tree, logging but swallowing OS errors."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to remove directory %s", path, exc_info=True)


def _write_all_bytes(file_obj: Any, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = file_obj.write(data[offset:])
        if written is None or written == 0:
            raise OSError("Failed to write complete payload to temporary file")
        offset += written


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_file_path = Path(tmp_path)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            _write_all_bytes(tmp_file, data)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        best_effort_remove(tmp_file_path)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically.

    Raw decoded API records with no fields removed or transformed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_file_path = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        best_effort_remove(tmp_file_path)
        raise
