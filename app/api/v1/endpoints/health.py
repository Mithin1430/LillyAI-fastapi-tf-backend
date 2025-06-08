"""
Health check endpoints
"""
from fastapi import APIRouter
from app.models.schemas import HealthResponse, RootResponse
from app.models.ml_models import model_manager
from app.config.settings import settings

router = APIRouter()

@router.get("/", response_model=RootResponse)
async def root():
    """Root endpoint with API information"""
    return RootResponse(
        message="LillyAI Character Recognition API",
        status="running",
        environment=settings.ENVIRONMENT,
        version=settings.API_VERSION,
        model_type="EMNIST ByClass",
        supported_characters=len(settings.EMNIST_BYCLASS_LABELS)
    )

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    model_status = "loaded" if model_manager.is_loaded() else "failed"
    model_info = model_manager.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model=model_status,
        model_info=model_info,
        config={
            "host": settings.HOST,
            "port": settings.PORT,
            "debug": settings.DEBUG,
            "training_data_path": settings.TRAINING_DATA_PATH,
            "environment": settings.ENVIRONMENT
        }
    )