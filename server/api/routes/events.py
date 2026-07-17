"""Idempotent edge event ingestion."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from server.api.dependencies import get_repository
from server.api.repositories import ControlPlaneRepository
from server.api.schemas import EventIngestResponse
from shared.protocol import EVENT_IDEMPOTENCY_HEADER
from shared.schemas import SafetyEvent

router = APIRouter(prefix="/events", tags=["events"])


@router.post("", response_model=EventIngestResponse)
def ingest_event(
    event: SafetyEvent,
    response: Response,
    repository: Annotated[ControlPlaneRepository, Depends(get_repository)],
    idempotency_key: Annotated[
        str | None,
        Header(alias=EVENT_IDEMPOTENCY_HEADER),
    ] = None,
) -> EventIngestResponse:
    if idempotency_key is not None and idempotency_key != str(event.event_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency-Key must equal event_id",
        )
    accepted = repository.ingest_event(event)
    response.status_code = (
        status.HTTP_201_CREATED if accepted else status.HTTP_200_OK
    )
    return EventIngestResponse(
        event_id=event.event_id,
        accepted=accepted,
        duplicate=not accepted,
    )
