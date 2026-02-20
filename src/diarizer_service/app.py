from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.diarizer_service.routes import engine, router as diarizer_router
from src.diarizer_service.schemas import HealthResponse


logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _start_background_warmup() -> None:
    def _warmup() -> None:
        t0 = time.time()
        try:
            logger.info("Diarizer warmup started (model download/load)")
            engine.load()
            logger.info(f"Diarizer warmup finished in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"Diarizer warmup failed (ignored): {e}")

    threading.Thread(target=_warmup, name="diarizer-warmup", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if _env_bool("DIARIZER_WARMUP_ON_STARTUP", default=False):
        _start_background_warmup()
    yield


app = FastAPI(title="TingWu Diarizer Service", version="1.0.0", lifespan=lifespan)

app.include_router(diarizer_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")
