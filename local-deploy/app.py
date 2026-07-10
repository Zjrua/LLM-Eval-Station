"""Standalone FastAPI app serving OMLX monitoring endpoints.

Run: uvicorn app:app --host 0.0.0.0 --port 8080
"""

from fastapi import FastAPI

from omlx_router import router as omlx_router

app = FastAPI(title="OMLX Monitor API", version="1.0.0")
app.include_router(omlx_router)


@app.get("/")
async def root():
    return {"service": "OMLX Monitor API", "version": "1.0.0"}
