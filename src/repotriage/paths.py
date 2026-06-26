"""Filesystem path-safety helpers shared across subsystems."""

from __future__ import annotations

from pathlib import Path


def resolve_within_directory(base_dir: Path, relative_path: str) -> Path:
    """Resolve a relative path against a base directory, rejecting escapes.

    Raises ``ValueError`` when ``relative_path`` is absolute or resolves to a
    location outside ``base_dir``. Callers wrap this into a domain-specific error.
    """
    if Path(relative_path).is_absolute():
        raise ValueError(f"Path must be relative: {relative_path!r}")

    candidate = (base_dir / relative_path).resolve()
    base_resolved = base_dir.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes base directory: {relative_path!r}") from exc
    return candidate
