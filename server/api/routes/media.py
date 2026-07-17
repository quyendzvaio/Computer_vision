"""Media upload contract; MinIO signing is intentionally deferred to Phase 4."""

from fastapi import APIRouter, HTTPException, status

from server.api.schemas import MediaPresignRequest

router = APIRouter(prefix="/media", tags=["media"])


@router.post("/presign")
def create_presigned_upload(_request: MediaPresignRequest) -> None:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="media signing adapter is not enabled in Phase 1",
    )
