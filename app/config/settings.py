"""
Application configuration settings
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings"""
    
    # API Information
    API_TITLE: str = "LillyAI Character Recognition API"
    API_DESCRIPTION: str = "AI-powered character recognition for educational games using EMNIST"
    API_VERSION: str = "2.0.0"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "modals/AlphaNumeric/model.h5")
    TRAINING_DATA_PATH: str = os.getenv("TRAINING_DATA_PATH", "Digits")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # EMNIST ByClass label mapping (0–61)
    EMNIST_BYCLASS_LABELS: List[str] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z'
    ]

# Create settings instance
settings = Settings()