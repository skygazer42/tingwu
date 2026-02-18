from __future__ import annotations

from fastapi import FastAPI

from src.diarizer_service.routes import router as diarizer_router
from src.diarizer_service.schemas import HealthResponse


app = FastAPI(title="TingWu Diarizer Service", version="1.0.0")

app.include_router(diarizer_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")

