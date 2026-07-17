"""Container liveness and readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from server.api.dependencies import get_repository
from server.api.repositories import ControlPlaneRepository
from server.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok", component="api")


@router.get("/health/ready", response_model=HealthResponse)
def ready(
    repository: Annotated[ControlPlaneRepository, Depends(get_repository)],
) -> HealthResponse:
    if not repository.healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="persistence is unavailable",
        )
    return HealthResponse(status="ready", component="api")
