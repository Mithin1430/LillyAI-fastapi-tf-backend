"""
Shared dependencies for the application
"""
from fastapi import HTTPException
from app.models.ml_models import model_manager

def get_model():
    """Dependency to get the ML model"""
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=500, 
            detail="EMNIST model not loaded. Please ensure model.h5 exists."
        )
    return model_manager.get_model()