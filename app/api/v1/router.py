"""
API v1 router configuration
"""
from fastapi import APIRouter
from app.api.v1.endpoints import health, prediction, training

# Create main v1 router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(prediction.router, tags=["prediction"])
api_router.include_router(training.router, tags=["training"])