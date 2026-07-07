"""Maintainer feedback route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from repotriage.api.dependencies import get_bundle, get_feedback_repository
from repotriage.api.schemas import FeedbackResponse
from repotriage.inference.artifact_loader import LoadedInferenceBundle
from repotriage.persistence.feedback_repository import FeedbackRepository
from repotriage.persistence.schemas import FeedbackRequest
from repotriage.persistence.validators import validate_feedback_request

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback_endpoint(
    body: FeedbackRequest,
    bundle: LoadedInferenceBundle = Depends(get_bundle),
    repository: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackResponse:
    """Store a maintainer review event for a prediction."""
    validate_feedback_request(body, bundle)
    return repository.store(body)
