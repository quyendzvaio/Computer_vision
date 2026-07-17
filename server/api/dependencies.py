"""FastAPI dependency providers."""

from fastapi import Request

from server.api.repositories import ControlPlaneRepository


def get_repository(request: Request) -> ControlPlaneRepository:
    return request.app.state.repository
