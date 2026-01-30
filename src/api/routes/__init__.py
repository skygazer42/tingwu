from fastapi import APIRouter
from src.api.routes.transcribe import router as transcribe_router

api_router = APIRouter()
api_router.include_router(transcribe_router)

__all__ = ['api_router']
