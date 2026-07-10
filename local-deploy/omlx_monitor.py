"""FastAPI router: OMLX server status monitoring endpoints.

GET /api/omlx/health  — liveness check
GET /api/omlx/models  — available model list
GET /api/omlx/status  — GPU/server status
GET /api/omlx/latency — round-trip latency test
"""

from fastapi import APIRouter

from omlx_client import check_health, get_models, get_status, measure_latency

router = APIRouter(prefix="/api/omlx", tags=["omlx"])


@router.get("/health")
async def omly_health():
    """Server health check — returns offline status rather than 500 on failure."""
    return await check_health()


@router.get("/models")
async def omly_models():
    """Fetch available models from OMLX /v1/models."""
    return await get_models()


@router.get("/status")
async def omly_status():
    """GPU and server status. Returns partial data if status endpoint is not exposed."""
    return await get_status()


@router.get("/latency")
async def omly_latency():
    """Measure 3-sample round-trip latency to OMLX."""
    return await measure_latency()
