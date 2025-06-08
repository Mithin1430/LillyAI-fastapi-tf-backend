"""
Training data endpoints
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.models.schemas import TrainingImageResponse, ErrorResponse
from app.services.training_service import training_service

router = APIRouter()

@router.post("/save_training_image", response_model=TrainingImageResponse)
async def save_training_image(
    file: UploadFile = File(...),
    predicted_digit: str = Form(...),
    actual_answer: str = Form(...),
    is_correct: str = Form(...)
):
    """
    Save training image with feedback
    
    Args:
        file: Uploaded image file
        predicted_digit: What the model predicted
        actual_answer: The correct answer
        is_correct: Whether the prediction was correct
        
    Returns:
        Success response with file path and character
    """
    try:
        # Convert is_correct to boolean
        is_correct_bool = is_correct.lower() == 'true'
        
        # Read file contents
        contents = await file.read()
        
        # Save training image
        file_path, target_character = training_service.save_training_image(
            contents, predicted_digit, actual_answer, is_correct_bool
        )
        
        return TrainingImageResponse(
            status="success",
            message="Training image saved successfully",
            file_path=file_path,
            character=target_character
        )
        
    except Exception as e:
        print("Error saving training image:", e)
        raise HTTPException(status_code=500, detail=str(e))