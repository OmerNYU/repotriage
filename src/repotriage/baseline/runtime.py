"""Numerical runtime introspection and controlled thread behavior for baseline builds.

This module is the single place that inspects the *loaded* numerical stack
(via threadpoolctl) and enforces a deterministic numerical thread limit around
fitting and scoring. Keeping it separate from ``models.py`` preserves that module
as pure schema + hashing with no runtime/backend imports.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from repotriage.baseline.models import (
    ENVIRONMENT_SCHEMA_VERSION,
    NUMERICAL_THREAD_LIMIT,
    BackendModule,
    EnvironmentMetadata,
    compute_numerical_environment_sha256,
)

_BACKEND_STABLE_FIELDS = (
    "user_api",
    "internal_api",
    "prefix",
    "version",
    "threading_layer",
    "architecture",
)


def _package_version(module_name: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(module_name)
    except Exception:
        return None


def collect_numerical_backends() -> list[BackendModule]:
    """Return the loaded numerical backends as canonical, sorted ``BackendModule``s.

    Imports NumPy, SciPy, and scikit-learn first so their bundled BLAS/OpenMP
    libraries are actually loaded before ``threadpool_info`` is queried. Only
    stable, output-relevant fields are retained; volatile fields (absolute
    ``filepath``, live ``num_threads``, PID, hostname) are dropped.
    """
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import threadpoolctl

    backends = [
        BackendModule(
            user_api=entry.get("user_api"),
            internal_api=entry.get("internal_api"),
            prefix=entry.get("prefix"),
            version=entry.get("version"),
            threading_layer=entry.get("threading_layer"),
            architecture=entry.get("architecture"),
        )
        for entry in threadpoolctl.threadpool_info()
    ]
    backends.sort(
        key=lambda backend: tuple(
            getattr(backend, field) or "" for field in _BACKEND_STABLE_FIELDS
        )
    )
    return backends


@contextmanager
def numerical_thread_limits(limit: int = NUMERICAL_THREAD_LIMIT) -> Iterator[None]:
    """Constrain all detected numerical backends to ``limit`` threads.

    Wrapping fitting and scoring in this context removes machine-default BLAS
    thread-count nondeterminism from generated model bytes and scores.
    """
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=limit):
        yield


def detect_blas_lapack_vendor() -> str | None:
    """Best-effort stable BLAS/LAPACK vendor string from NumPy build config."""
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        libraries = config.get("Build Dependencies", {})
        blas = libraries.get("blas")
        lapack = libraries.get("lapack")
        blas_name = blas.get("name") if isinstance(blas, dict) else (
            str(blas) if blas is not None else None
        )
        lapack_name = lapack.get("name") if isinstance(lapack, dict) else (
            str(lapack) if lapack is not None else None
        )
        parts = [part for part in (blas_name, lapack_name) if part]
        return "+".join(parts) if parts else None
    except Exception:
        return None


def build_environment_fingerprint(
    *,
    numpy_version: str | None = None,
    scipy_version: str | None = None,
    scikit_learn_version: str | None = None,
    joblib_version: str | None = None,
    thread_limit: int = NUMERICAL_THREAD_LIMIT,
) -> tuple[EnvironmentMetadata, str]:
    """Collect environment metadata and its numerical fingerprint hash."""
    backends = collect_numerical_backends()
    env = EnvironmentMetadata(
        python_implementation=platform.python_implementation(),
        python_version=sys.version,
        os_system=platform.system(),
        platform=platform.platform(),
        machine_architecture=platform.machine(),
        numpy_version=numpy_version,
        scipy_version=scipy_version,
        scikit_learn_version=scikit_learn_version,
        joblib_version=joblib_version,
        threadpoolctl_version=_package_version("threadpoolctl"),
        blas_lapack_vendor=detect_blas_lapack_vendor(),
        numerical_backends=backends,
        numerical_thread_limit=thread_limit,
        reproducibility_note=(
            "Metrics and model outputs are reproducible within the recorded numerical "
            "environment fingerprint (interpreter, package versions, exact numerical "
            "backend versions, and a controlled numerical thread limit). Identical "
            "experiment hashes may have multiple environment-specific runs. Numerical "
            "equivalence across environments is not guaranteed."
        ),
        serialization_security_warning=(
            "model.joblib uses pickle-based serialization. Do not load model files from "
            "untrusted sources."
        ),
    )
    env_hash = compute_numerical_environment_sha256(
        environment_schema_version=ENVIRONMENT_SCHEMA_VERSION,
        python_implementation=env.python_implementation,
        python_version=env.python_version,
        os_system=env.os_system,
        machine_architecture=env.machine_architecture,
        numpy_version=env.numpy_version,
        scipy_version=env.scipy_version,
        scikit_learn_version=env.scikit_learn_version,
        joblib_version=env.joblib_version,
        threadpoolctl_version=env.threadpoolctl_version,
        blas_lapack_vendor=env.blas_lapack_vendor,
        numerical_backends=env.numerical_backends,
        numerical_thread_limit=env.numerical_thread_limit,
    )
    return env, env_hash
