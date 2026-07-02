"""Tests for the numerical-environment fingerprint and controlled thread behavior."""

from __future__ import annotations

from repotriage.baseline import models_ml, runtime
from repotriage.baseline.models import (
    ENVIRONMENT_SCHEMA_VERSION,
    NUMERICAL_THREAD_LIMIT,
    BackendModule,
    canonical_backend_entries,
    compute_numerical_environment_sha256,
)


def _env_hash(
    *,
    backends: list[BackendModule],
    thread_limit: int = NUMERICAL_THREAD_LIMIT,
    numpy_version: str = "2.4.4",
) -> str:
    return compute_numerical_environment_sha256(
        environment_schema_version=ENVIRONMENT_SCHEMA_VERSION,
        python_implementation="CPython",
        python_version="3.13.2",
        os_system="Linux",
        machine_architecture="x86_64",
        numpy_version=numpy_version,
        scipy_version="1.17.1",
        scikit_learn_version="1.8.0",
        joblib_version="1.5.3",
        threadpoolctl_version="3.6.0",
        blas_lapack_vendor="scipy-openblas",
        numerical_backends=backends,
        numerical_thread_limit=thread_limit,
    )


def _backend(version: str, *, prefix: str = "libscipy_openblas") -> BackendModule:
    return BackendModule(
        user_api="blas",
        internal_api="openblas",
        prefix=prefix,
        version=version,
        threading_layer="pthreads",
        architecture="Haswell",
    )


def test_exact_backend_version_changes_env_hash() -> None:
    a = _env_hash(backends=[_backend("0.3.30")])
    b = _env_hash(backends=[_backend("0.3.33")])
    assert a != b


def test_backend_entry_ordering_does_not_change_env_hash() -> None:
    b1 = _backend("0.3.30", prefix="libscipy_openblas")
    b2 = _backend("0.3.33", prefix="libgomp")
    forward = _env_hash(backends=[b1, b2])
    reversed_order = _env_hash(backends=[b2, b1])
    assert forward == reversed_order


def test_thread_limit_changes_env_hash() -> None:
    one = _env_hash(backends=[_backend("0.3.30")], thread_limit=1)
    two = _env_hash(backends=[_backend("0.3.30")], thread_limit=2)
    assert one != two


def test_package_version_changes_env_hash() -> None:
    a = _env_hash(backends=[_backend("0.3.30")], numpy_version="2.4.4")
    b = _env_hash(backends=[_backend("0.3.30")], numpy_version="2.5.0")
    assert a != b


def test_canonical_backend_entries_drops_volatile_fields() -> None:
    backend = BackendModule(
        user_api="blas",
        internal_api="openblas",
        prefix="libscipy_openblas",
        version="0.3.30",
        threading_layer="pthreads",
        architecture="Haswell",
    )
    entries = canonical_backend_entries([backend])
    assert entries == [
        {
            "architecture": "Haswell",
            "internal_api": "openblas",
            "prefix": "libscipy_openblas",
            "threading_layer": "pthreads",
            "user_api": "blas",
            "version": "0.3.30",
        }
    ]
    assert "filepath" not in entries[0]
    assert "num_threads" not in entries[0]


def test_build_environment_fingerprint_includes_platform_and_versions() -> None:
    env, env_hash = runtime.build_environment_fingerprint(
        numpy_version="2.4.4",
        scipy_version="1.17.1",
        scikit_learn_version="1.8.0",
        joblib_version="1.5.3",
    )
    assert env.python_implementation
    assert env.python_version
    assert env.os_system
    assert env.platform
    assert env.machine_architecture
    assert env.numpy_version == "2.4.4"
    assert env.numerical_thread_limit == NUMERICAL_THREAD_LIMIT
    assert env.numerical_backends  # backends loaded after importing numpy/scipy/sklearn
    assert len(env_hash) == 64


def test_build_environment_fingerprint_thread_limit_binds_identity() -> None:
    _env_one, hash_one = runtime.build_environment_fingerprint(
        numpy_version="2.4.4", thread_limit=1
    )
    _env_two, hash_two = runtime.build_environment_fingerprint(
        numpy_version="2.4.4", thread_limit=2
    )
    assert hash_one != hash_two


def test_training_uses_declared_thread_limit(monkeypatch) -> None:
    import numpy as np

    from repotriage.baseline.config import load_baseline_config
    from tests.helpers import write_baseline_config

    entered: list[str] = []

    class _SpyContext:
        def __enter__(self):
            entered.append("enter")
            return self

        def __exit__(self, *exc):
            return False

    def _spy(limit: int = NUMERICAL_THREAD_LIMIT):
        entered.append(f"limit={limit}")
        return _SpyContext()

    monkeypatch.setattr(models_ml, "numerical_thread_limits", _spy)

    config, _, _, _ = load_baseline_config(write_baseline_config(np_tmp := _tmp()))
    candidate = config.candidates[0]
    labels = ["Bug", "Docs"]
    train_texts = ["bug crash", "documentation fix", "another bug", "more docs"]
    train_targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int8)
    model, _report = models_ml.train_model(
        candidate=candidate,
        labels=labels,
        train_texts=train_texts,
        train_targets=train_targets,
        random_state=42,
        threshold=0.5,
    )
    assert "enter" in entered
    assert any(item == f"limit={NUMERICAL_THREAD_LIMIT}" for item in entered)

    entered.clear()
    model.predict_proba_matrix(train_texts)
    assert "enter" in entered


def _tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp()) / "baseline.json"
