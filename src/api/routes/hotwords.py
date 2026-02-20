"""热词管理 API"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.engine import transcription_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hotwords", tags=["hotwords"])


class HotwordsListResponse(BaseModel):
    code: int = 0
    hotwords: List[str] = Field(..., description="热词列表")
    count: int = Field(..., description="热词数量")


class HotwordsUpdateRequest(BaseModel):
    hotwords: List[str] = Field(..., description="热词列表")


class HotwordsUpdateResponse(BaseModel):
    code: int = 0
    count: int = Field(..., description="更新后的热词数量")
    message: str = "success"


@router.get("", response_model=HotwordsListResponse)
async def get_hotwords():
    """获取当前热词列表"""
    hotwords = list(transcription_engine.corrector.hotwords.keys())
    return HotwordsListResponse(
        hotwords=hotwords,
        count=len(hotwords)
    )


@router.get("/context", response_model=HotwordsListResponse)
async def get_context_hotwords():
    """获取当前上下文热词列表（仅用于注入提示，不做强制替换）"""
    hotwords = list(getattr(transcription_engine, "_context_hotwords_list", []) or [])
    # Defensive: keep only non-empty strings.
    hotwords = [str(h).strip() for h in hotwords if str(h).strip()]
    return HotwordsListResponse(
        hotwords=hotwords,
        count=len(hotwords),
    )


@router.post("", response_model=HotwordsUpdateResponse)
async def update_hotwords(request: HotwordsUpdateRequest):
    """更新热词列表 (替换全部)"""
    try:
        transcription_engine.update_hotwords(request.hotwords)
        return HotwordsUpdateResponse(
            count=len(request.hotwords),
            message="热词更新成功"
        )
    except Exception as e:
        logger.error(f"Failed to update hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context", response_model=HotwordsUpdateResponse)
async def update_context_hotwords(request: HotwordsUpdateRequest):
    """更新上下文热词列表 (替换全部)"""
    try:
        transcription_engine.update_context_hotwords(request.hotwords)
        return HotwordsUpdateResponse(
            count=len(request.hotwords),
            message="上下文热词更新成功",
        )
    except Exception as e:
        logger.error(f"Failed to update context hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/append", response_model=HotwordsUpdateResponse)
async def append_hotwords(request: HotwordsUpdateRequest):
    """追加热词 (保留现有)"""
    try:
        existing = list(transcription_engine.corrector.hotwords.keys())
        combined = list(set(existing + request.hotwords))
        transcription_engine.update_hotwords(combined)
        return HotwordsUpdateResponse(
            count=len(combined),
            message=f"追加了 {len(request.hotwords)} 个热词"
        )
    except Exception as e:
        logger.error(f"Failed to append hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/append", response_model=HotwordsUpdateResponse)
async def append_context_hotwords(request: HotwordsUpdateRequest):
    """追加上下文热词 (保留现有)"""
    try:
        existing = list(getattr(transcription_engine, "_context_hotwords_list", []) or [])
        existing_norm = [str(h).strip() for h in existing if str(h).strip()]
        existing_set = set(existing_norm)

        to_append = []
        for h in request.hotwords:
            s = str(h).strip()
            if not s:
                continue
            if s in existing_set:
                continue
            existing_set.add(s)
            to_append.append(s)

        combined = existing_norm + to_append
        transcription_engine.update_context_hotwords(combined)
        return HotwordsUpdateResponse(
            count=len(combined),
            message=f"追加了 {len(to_append)} 个上下文热词",
        )
    except Exception as e:
        logger.error(f"Failed to append context hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload", response_model=HotwordsUpdateResponse)
async def reload_hotwords():
    """从文件重新加载热词"""
    try:
        transcription_engine.load_hotwords()
        count = len(transcription_engine.corrector.hotwords)
        return HotwordsUpdateResponse(
            count=count,
            message="热词重新加载成功"
        )
    except Exception as e:
        logger.error(f"Failed to reload hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/reload", response_model=HotwordsUpdateResponse)
async def reload_context_hotwords():
    """从文件重新加载上下文热词"""
    try:
        transcription_engine.load_context_hotwords()
        count = len(getattr(transcription_engine, "_context_hotwords_list", []) or [])
        return HotwordsUpdateResponse(
            count=count,
            message="上下文热词重新加载成功",
        )
    except Exception as e:
        logger.error(f"Failed to reload context hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))
