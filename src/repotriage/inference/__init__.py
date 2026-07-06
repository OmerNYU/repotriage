"""Local artifact-backed issue inference for RepoTriage."""

from repotriage.inference.models import InferenceError
from repotriage.inference.pipeline import infer_issue

__all__ = ["InferenceError", "infer_issue"]
