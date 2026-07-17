"""Control-plane resource lifecycle."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.api.repositories import InMemoryControlPlaneRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not hasattr(app.state, "repository"):
        app.state.repository = InMemoryControlPlaneRepository()
    yield
