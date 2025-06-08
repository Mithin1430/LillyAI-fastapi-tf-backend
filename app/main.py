"""
FastAPI application main module
"""
from fastapi import FastAPI
from app.config.settings import settings
from app.core.middleware import setup_cors_middleware
from app.api.v1.router import api_router

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Setup middleware
setup_cors_middleware(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# For backward compatibility, also include routes at root level
app.include_router(api_router)