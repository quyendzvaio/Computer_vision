"""Device heartbeat, metrics and last-known configuration endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from server.api.dependencies import get_repository
from server.api.repositories import ControlPlaneRepository
from server.api.schemas import ConfigurationResponse
from shared.schemas import DeviceHeartbeat, DeviceMetrics

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/{device_id}/heartbeat", status_code=status.HTTP_202_ACCEPTED)
def receive_heartbeat(
    device_id: str,
    heartbeat: DeviceHeartbeat,
    repository: Annotated[ControlPlaneRepository, Depends(get_repository)],
) -> dict[str, str]:
    repository.save_heartbeat(device_id, heartbeat)
    return {"status": "accepted"}


@router.post("/{device_id}/metrics", status_code=status.HTTP_202_ACCEPTED)
def receive_metrics(
    device_id: str,
    metrics: DeviceMetrics,
    repository: Annotated[ControlPlaneRepository, Depends(get_repository)],
) -> dict[str, str]:
    repository.save_metrics(device_id, metrics)
    return {"status": "accepted"}


@router.get("/{device_id}/configuration", response_model=ConfigurationResponse)
def get_configuration(
    device_id: str,
    repository: Annotated[ControlPlaneRepository, Depends(get_repository)],
) -> ConfigurationResponse:
    configuration = repository.get_configuration(device_id)
    if configuration is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no configuration published for device",
        )
    return configuration
