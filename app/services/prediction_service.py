"""
Prediction service for character recognition
"""
from PIL import Image
import numpy as np
from typing import Tuple
from app.models.ml_models import model_manager
from app.utils.image_processing import preprocess_image_for_prediction
from app.config.settings import settings

class PredictionService:
    """Service for handling character predictions"""
    
    @staticmethod
    def predict_character_from_image(img: Image.Image) -> Tuple[int, str, float]:
        """
        Predict character from image using EMNIST model
        
        Args:
            img: PIL Image object
            
        Returns:
            Tuple of (label_index, predicted_character, confidence)
        """
        try:
            # Check if model is loaded
            if not model_manager.is_loaded():
                raise Exception("EMNIST model not loaded")
            
            # Preprocess image
            img_array = preprocess_image_for_prediction(img)
            
            # Get predictions
            model = model_manager.get_model()
            predictions = model.predict(img_array, verbose=0)
            label_index = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            predicted_char = settings.EMNIST_BYCLASS_LABELS[label_index]
            
            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Top 5 predictions: {np.argsort(predictions[0])[-5:][::-1]}")
            print(f"Predicted character: '{predicted_char}' (index: {label_index}), Confidence: {confidence:.3f}")
            
            return label_index, predicted_char, confidence
            
        except Exception as e:
            print(f"Error in character prediction: {e}")
            raise e
    
    @staticmethod
    def format_prediction_response(predicted_char: str, label_index: int, confidence: float) -> dict:
        """
        Format prediction response for API
        
        Args:
            predicted_char: Predicted character
            label_index: Label index
            confidence: Prediction confidence
            
        Returns:
            Formatted response dictionary
        """
        # For backward compatibility with frontend expecting "prediction" field
        # If it's a digit (0-9), return the numeric value, otherwise return the character
        if predicted_char.isdigit():
            prediction_value = int(predicted_char)
        else:
            prediction_value = predicted_char

        return {
            "prediction": prediction_value,
            "label": label_index,
            "character": predicted_char,
            "confidence": confidence,
            "status": "success"
        }

# Service instance
prediction_service = PredictionService()