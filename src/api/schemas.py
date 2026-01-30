"""API 请求/响应模式"""
from typing import List, Optional
from pydantic import BaseModel, Field


class SentenceInfo(BaseModel):
    """句子信息"""
    text: str = Field(..., description="句子文本")
    start: int = Field(..., description="开始时间 (毫秒)")
    end: int = Field(..., description="结束时间 (毫秒)")
    speaker: Optional[str] = Field(default=None, description="说话人标签")
    speaker_id: Optional[int] = Field(default=None, description="说话人 ID")


class TranscribeResponse(BaseModel):
    """转写响应"""
    code: int = Field(default=0, description="状态码 (0=成功)")
    text: str = Field(..., description="完整转写文本")
    sentences: List[SentenceInfo] = Field(default=[], description="分句信息")
    transcript: Optional[str] = Field(default=None, description="格式化转写稿")
    raw_text: Optional[str] = Field(default=None, description="原始文本 (未纠错)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
