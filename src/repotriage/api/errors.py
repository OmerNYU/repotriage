"""Map inference domain exceptions to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from repotriage.inference.models import InferenceError, InferenceInputError


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for inference domain errors."""

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
