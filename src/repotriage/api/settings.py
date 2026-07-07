"""API server settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repotriage.baseline.builder import DEFAULT_BASELINES_ROOT
from repotriage.inference.artifact_loader import (
    DEFAULT_ABSTENTION_POLICIES_ROOT,
    DEFAULT_RETRIEVAL_BASELINES_ROOT,
)
from repotriage.model_dataset.builder import DEFAULT_MODEL_READY_ROOT
from repotriage.threshold_policy.builder import DEFAULT_THRESHOLD_POLICIES_ROOT

INFERENCE_CONFIG_ENV_VAR = "REPOTRIAGE_INFERENCE_CONFIG"


@dataclass(frozen=True)
class ApiSettings:
    """Runtime configuration for the inference API server."""

    inference_config_path: Path
    baselines_root: Path = DEFAULT_BASELINES_ROOT
    threshold_policies_root: Path = DEFAULT_THRESHOLD_POLICIES_ROOT
    abstention_policies_root: Path = DEFAULT_ABSTENTION_POLICIES_ROOT
    retrieval_baselines_root: Path = DEFAULT_RETRIEVAL_BASELINES_ROOT
    model_ready_root: Path = DEFAULT_MODEL_READY_ROOT

    @classmethod
    def from_env(cls) -> ApiSettings:
        """Load settings from REPOTRIAGE_INFERENCE_CONFIG."""
        raw = os.environ.get(INFERENCE_CONFIG_ENV_VAR)
        if not raw:
            raise ValueError(
                f"{INFERENCE_CONFIG_ENV_VAR} environment variable is required "
                "when settings are not passed explicitly."
            )
        return cls(inference_config_path=Path(raw))

    @classmethod
    def from_namespace(cls, args: Any) -> ApiSettings:
        """Build settings from a CLI argparse namespace."""
        return cls(
            inference_config_path=args.config,
            baselines_root=args.baselines_root,
            threshold_policies_root=args.threshold_policies_root,
            abstention_policies_root=args.abstention_policies_root,
            retrieval_baselines_root=args.retrieval_baselines_root,
            model_ready_root=args.model_ready_root,
        )
