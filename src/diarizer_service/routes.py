from __future__ import annotations

import io
import wave

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.diarizer_service.schemas import DiarizeResponse


router = APIRouter(prefix="/api/v1", tags=["diarizer"])


@router.post("/diarize", response_model=DiarizeResponse)
async def diarize(file: UploadFile = File(...)) -> DiarizeResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="missing audio file")

    # Validate WAV container quickly. The TingWu client always uploads a WAV.
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            _ = wf.getnframes()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid wav: {e}")

    # Stub response (real engine wiring comes in later tasks).
    return DiarizeResponse(segments=[])

