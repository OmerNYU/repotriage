"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from repotriage.api.dependencies import get_bundle
from repotriage.api.schemas import HealthResponse
from repotriage.inference.artifact_loader import LoadedInferenceBundle

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(bundle: LoadedInferenceBundle = Depends(get_bundle)) -> HealthResponse:
    """Return liveness status and loaded bundle metadata."""
    return HealthResponse(
        repository=bundle.repository.full_name,
        inference_config_path=str(bundle.config_path),
    )
