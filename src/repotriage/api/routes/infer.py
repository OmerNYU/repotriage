"""Issue inference route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from repotriage.api.dependencies import get_bundle
from repotriage.api.schemas import InferRequest
from repotriage.inference.artifact_loader import LoadedInferenceBundle
from repotriage.inference.models import InferenceResponse
from repotriage.inference.pipeline import infer_issue

router = APIRouter(tags=["inference"])


@router.post("/infer", response_model=InferenceResponse)
def infer_endpoint(
    body: InferRequest,
    bundle: LoadedInferenceBundle = Depends(get_bundle),
) -> InferenceResponse:
    """Score a new issue using the loaded inference bundle."""
    return infer_issue(bundle, body)
