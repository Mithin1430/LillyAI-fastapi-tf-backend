"""
Custom middleware for the application
"""
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings

def setup_cors_middleware(app):
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,    
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )