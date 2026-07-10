"""Async HTTP client for OMLX server (192.168.192.2:8000) via ZeroTier."""

import asyncio
import time
from typing import Optional

import httpx


OMLX_BASE_URL = "http://192.168.192.2:8000"
OMLX_API_KEY = "1203"
REQUEST_TIMEOUT = 10.0  # seconds


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {OMLX_API_KEY}"}


async def _get(path: str, timeout: float = REQUEST_TIMEOUT) -> httpx.Response:
    """GET request to OMLX. Caller must handle errors."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(f"{OMLX_BASE_URL}{path}", headers=_headers())


async def check_health() -> dict:
    """Health check — attempt to reach /v1/models as liveness probe."""
    try:
        start = time.monotonic()
        resp = await _get("/v1/models", timeout=5.0)
        elapsed = time.monotonic() - start
        return {
            "status": "online",
            "reachable": True,
            "response_ms": round(elapsed * 1000, 1),
            "http_status": resp.status_code,
        }
    except httpx.ConnectError:
        return {"status": "offline", "reachable": False, "error": "connection_refused"}
    except httpx.ConnectTimeout:
        return {"status": "offline", "reachable": False, "error": "connection_timeout"}
    except httpx.ReadTimeout:
        return {"status": "degraded", "reachable": True, "error": "read_timeout"}
    except Exception as exc:
        return {"status": "offline", "reachable": False, "error": str(exc)}


async def get_models() -> dict:
    """Fetch available models from /v1/models."""
    try:
        resp = await _get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        models = [
            {
                "id": m.get("id", "unknown"),
                "owned_by": m.get("owned_by", "unknown"),
            }
            for m in data.get("data", [])
        ]
        return {"status": "online", "count": len(models), "models": models}
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return {"status": "offline", "count": 0, "models": [], "error": "server_unreachable"}
    except httpx.ReadTimeout:
        return {"status": "degraded", "count": 0, "models": [], "error": "read_timeout"}
    except Exception as exc:
        return {"status": "error", "count": 0, "models": [], "error": str(exc)}


async def get_status() -> dict:
    """Fetch GPU/server status. OMLX may or may not expose a status endpoint."""
    # Try common vLLM/OMLX status paths
    for path in ["/status", "/v1/status", "/health"]:
        try:
            resp = await _get(path, timeout=5.0)
            if resp.status_code == 200:
                return {"status": "online", "endpoint": path, "data": resp.json()}
            if resp.status_code == 404:
                continue  # try next path
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return {"status": "offline", "error": "server_unreachable"}
        except Exception:
            continue

    # If no status endpoint found, report what we know
    health = await check_health()
    if health["status"] == "online":
        return {
            "status": "online",
            "gpu_info": None,
            "note": "No dedicated status endpoint exposed by OMLX. Server is reachable.",
            "server_health": health,
        }
    return {"status": "offline", "error": "server_unreachable"}


async def measure_latency() -> dict:
    """Measure round-trip latency to OMLX /v1/models endpoint."""
    samples = []
    for _ in range(3):
        try:
            start = time.monotonic()
            resp = await _get("/v1/models", timeout=10.0)
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            samples.append({"ms": elapsed_ms, "ok": resp.status_code < 500})
        except (httpx.ConnectError, httpx.ConnectTimeout):
            samples.append({"ms": None, "ok": False, "error": "unreachable"})
        except Exception as exc:
            samples.append({"ms": None, "ok": False, "error": str(exc)})

    reachable = [s for s in samples if s["ok"] and s["ms"] is not None]
    if reachable:
        latencies = [s["ms"] for s in reachable]
        return {
            "status": "online",
            "samples": samples,
            "avg_ms": round(sum(latencies) / len(latencies), 1),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
        }
    return {"status": "offline", "samples": samples, "error": "no_successful_requests"}
