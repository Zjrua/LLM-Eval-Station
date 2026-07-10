"""Tests for OMLX monitoring API endpoints."""

import sys
import os

# Ensure the workspace dir is on sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from httpx import AsyncClient, ASGITransport

from app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "OMLX Monitor API"


@pytest.mark.anyio
async def test_health_returns_offline(client, monkeypatch):
    """Health endpoint returns offline status when OMLX is unreachable,
    never a 500 error."""
    async def mock_check():
        return {"status": "offline", "reachable": False, "error": "connection_refused"}

    monkeypatch.setattr("omlx_router.check_health", mock_check)
    resp = await client.get("/api/omlx/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "offline"
    assert not data["reachable"]


@pytest.mark.anyio
async def test_health_returns_online(client, monkeypatch):
    async def mock_check():
        return {"status": "online", "reachable": True, "response_ms": 45.2, "http_status": 200}

    monkeypatch.setattr("omlx_router.check_health", mock_check)
    resp = await client.get("/api/omlx/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["response_ms"] == 45.2


@pytest.mark.anyio
async def test_models_returns_offline(client, monkeypatch):
    async def mock_get():
        return {"status": "offline", "count": 0, "models": [], "error": "server_unreachable"}

    monkeypatch.setattr("omlx_router.get_models", mock_get)
    resp = await client.get("/api/omlx/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "offline"
    assert data["count"] == 0
    assert data["models"] == []


@pytest.mark.anyio
async def test_models_returns_list(client, monkeypatch):
    async def mock_get():
        return {
            "status": "online",
            "count": 2,
            "models": [
                {"id": "Qwen3.5-9B", "owned_by": "alibaba"},
                {"id": "Qwen3.6-27B", "owned_by": "alibaba"},
            ],
        }

    monkeypatch.setattr("omlx_router.get_models", mock_get)
    resp = await client.get("/api/omlx/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["count"] == 2
    assert len(data["models"]) == 2
    assert data["models"][0]["id"] == "Qwen3.5-9B"


@pytest.mark.anyio
async def test_status_offline(client, monkeypatch):
    async def mock_get():
        return {"status": "offline", "error": "server_unreachable"}

    monkeypatch.setattr("omlx_router.get_status", mock_get)
    resp = await client.get("/api/omlx/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "offline"


@pytest.mark.anyio
async def test_latency_offline(client, monkeypatch):
    async def mock_measure():
        return {
            "status": "offline",
            "samples": [
                {"ms": None, "ok": False, "error": "unreachable"},
                {"ms": None, "ok": False, "error": "unreachable"},
                {"ms": None, "ok": False, "error": "unreachable"},
            ],
            "error": "no_successful_requests",
        }

    monkeypatch.setattr("omlx_router.measure_latency", mock_measure)
    resp = await client.get("/api/omlx/latency")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "offline"
    assert len(data["samples"]) == 3


@pytest.mark.anyio
async def test_latency_online(client, monkeypatch):
    async def mock_measure():
        return {
            "status": "online",
            "samples": [
                {"ms": 42.1, "ok": True},
                {"ms": 38.7, "ok": True},
                {"ms": 51.3, "ok": True},
            ],
            "avg_ms": 44.0,
            "min_ms": 38.7,
            "max_ms": 51.3,
        }

    monkeypatch.setattr("omlx_router.measure_latency", mock_measure)
    resp = await client.get("/api/omlx/latency")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["avg_ms"] == 44.0
    assert data["min_ms"] == 38.7
