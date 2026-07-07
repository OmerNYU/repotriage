"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from repotriage.api.errors import register_exception_handlers
from repotriage.api.routes.health import router as health_router
from repotriage.api.routes.infer import router as infer_router
from repotriage.api.settings import ApiSettings
from repotriage.github.models import parse_repository
from repotriage.inference.artifact_loader import LoadedInferenceBundle, load_inference_bundle
from repotriage.inference.config import load_inference_config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def create_app(
    *,
    settings: ApiSettings | None = None,
    bundle: LoadedInferenceBundle | None = None,
) -> FastAPI:
    """Create a FastAPI application with the inference bundle loaded at startup."""
    resolved_settings = settings
    if resolved_settings is None and bundle is None:
        resolved_settings = ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.preloaded_bundle is not None:
            app.state.bundle = app.state.preloaded_bundle
        else:
            assert app.state.settings is not None
            config = load_inference_config(app.state.settings.inference_config_path)
            repository = parse_repository(config.repository)
            app.state.bundle = load_inference_bundle(
                app.state.settings.inference_config_path,
                repository=repository,
                baselines_root=app.state.settings.baselines_root,
                threshold_policies_root=app.state.settings.threshold_policies_root,
                abstention_policies_root=app.state.settings.abstention_policies_root,
                retrieval_baselines_root=app.state.settings.retrieval_baselines_root,
                model_ready_root=app.state.settings.model_ready_root,
            )
        yield

    app = FastAPI(
        title="RepoTriage Inference API",
        version="1",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.preloaded_bundle = bundle
    app.state.bundle = None

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(infer_router, prefix="/api/v1")

    return app
