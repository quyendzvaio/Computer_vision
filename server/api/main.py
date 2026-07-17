"""CPU-only FastAPI application factory."""

from fastapi import FastAPI

from server.api.lifespan import lifespan
from server.api.repositories import ControlPlaneRepository
from server.api.routes import devices, events, health, media
from shared.protocol import API_PREFIX, PROTOCOL_VERSION


def create_app(repository: ControlPlaneRepository | None = None) -> FastAPI:
    app = FastAPI(
        title="Construction Safety Control Plane",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    if repository is not None:
        app.state.repository = repository
    app.include_router(health.router)
    app.include_router(events.router, prefix=API_PREFIX)
    app.include_router(devices.router, prefix=API_PREFIX)
    app.include_router(media.router, prefix=API_PREFIX)
    return app


app = create_app()
