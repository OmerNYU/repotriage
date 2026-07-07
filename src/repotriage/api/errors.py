"""Map inference and persistence domain exceptions to HTTP responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from repotriage.inference.models import InferenceError, InferenceInputError
from repotriage.persistence.errors import FeedbackValidationError, PersistenceError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for inference and persistence domain errors."""

    @app.exception_handler(InferenceInputError)
    async def handle_inference_input_error(
        _request: Request,
        exc: InferenceInputError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(InferenceError)
    async def handle_inference_error(
        _request: Request,
        exc: InferenceError,
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(FeedbackValidationError)
    async def handle_feedback_validation_error(
        _request: Request,
        exc: FeedbackValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PersistenceError)
    async def handle_persistence_error(
        _request: Request,
        exc: PersistenceError,
    ) -> JSONResponse:
        logger.exception("Feedback persistence failed", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Failed to store feedback."})
