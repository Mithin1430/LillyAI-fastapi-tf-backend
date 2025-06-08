"""
Prediction endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from app.models.schemas import PredictionResponse, ErrorResponse
from app.services.prediction_service import prediction_service
from app.models.ml_models import model_manager

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict character from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with character, confidence, etc.
    """
    try:
        # Check if model is loaded
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=500,
                detail="EMNIST model not loaded. Please ensure model.h5 exists."
            )
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image."
            )
        
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Get prediction
        label_index, predicted_char, confidence = prediction_service.predict_character_from_image(img)
        
        # Format response
        response_data = prediction_service.format_prediction_response(
            predicted_char, label_index, confidence
        )
        
        return PredictionResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))