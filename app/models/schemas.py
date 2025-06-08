"""
Pydantic models for request/response schemas
"""
from pydantic import BaseModel
from typing import Optional, List, Union

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model: str
    model_info: dict
    config: dict

class RootResponse(BaseModel):
    """Root endpoint response schema"""
    message: str
    status: str
    environment: str
    version: str
    model_type: str
    supported_characters: int

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    prediction: Union[int, str]
    label: int
    character: str
    confidence: float
    status: str

class TrainingImageResponse(BaseModel):
    """Training image save response schema"""
    status: str
    message: str
    file_path: str
    character: str

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    status: str