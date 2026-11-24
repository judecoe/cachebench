from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness only: the process is up and serving. Dependency checks come later."""
    return {"status": "ok"}
