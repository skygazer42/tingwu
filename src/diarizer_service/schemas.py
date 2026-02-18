from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")


class DiarizeSegment(BaseModel):
    spk: int = Field(..., description="Speaker ID (0-based)")
    start: int = Field(..., description="Start time (ms)")
    end: int = Field(..., description="End time (ms)")


class DiarizeResponse(BaseModel):
    segments: List[DiarizeSegment] = Field(default_factory=list, description="Raw diarization segments")
    duration_ms: Optional[int] = Field(default=None, description="Audio duration (ms), when known")
    speakers: Optional[int] = Field(default=None, description="Number of unique speakers, when known")

