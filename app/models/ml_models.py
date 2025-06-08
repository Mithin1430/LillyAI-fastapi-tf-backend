"""
Machine Learning model management
"""
import tensorflow as tf
import os
from app.config.settings import settings

class ModelManager:
    """Manages ML model loading and access"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the EMNIST model"""
        try:
            self.model = tf.keras.models.load_model(settings.MODEL_PATH)
            print(f"EMNIST model loaded successfully from: {settings.MODEL_PATH}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            print(f"Model supports {len(settings.EMNIST_BYCLASS_LABELS)} character classes")
        except Exception as e:
            print(f"Error loading model from {settings.MODEL_PATH}: {e}")
            print("Make sure you have the model.h5 file in the modals/AlphaNumeric/ directory")
            print(f"Expected path: {os.path.abspath(settings.MODEL_PATH)}")
            self.model = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model(self):
        """Get the loaded model"""
        return self.model
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if self.model is None:
            return {}
        
        return {
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "model_path": settings.MODEL_PATH,
            "model_type": "EMNIST ByClass",
            "supported_characters": len(settings.EMNIST_BYCLASS_LABELS),
            "character_classes": settings.EMNIST_BYCLASS_LABELS[:10] + ["..."] + settings.EMNIST_BYCLASS_LABELS[-10:]
        }

# Global model manager instance
model_manager = ModelManager()