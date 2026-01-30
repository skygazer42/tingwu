"""TingWu Speech Service 主入口"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api import api_router
from src.api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    # 加载热词 (延迟加载模型)
    from src.core.engine import transcription_engine
    transcription_engine.load_hotwords()

    logger.info("Service ready!")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="基于 FunASR + CapsWriter 的中文语音转写服务",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """健康检查"""
    return HealthResponse(status="healthy", version=settings.version)


@app.get("/", tags=["system"])
async def root():
    """服务信息"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
    }
