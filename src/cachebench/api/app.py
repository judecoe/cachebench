"""FastAPI application factory.

Run with a single worker only (`uvicorn ... --factory --workers 1`): the semantic
index and the coalescing map live in process memory. See docs/design.md, "Process model".
"""

from fastapi import FastAPI

from cachebench import __version__
from cachebench.api.routes import router
from cachebench.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="CacheBench", version=__version__)
    app.state.settings = settings or Settings()
    app.include_router(router)
    return app
