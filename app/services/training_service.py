"""
Training service for handling training data
"""
import os
import uuid
from typing import Tuple
from app.config.settings import settings

class TrainingService:
    """Service for handling training data operations"""
    
    @staticmethod
    def determine_target_folder(actual_answer: str, is_correct: bool) -> str:
        """
        Determine the target folder for saving training image
        
        Args:
            actual_answer: The actual/correct answer
            is_correct: Whether the prediction was correct
            
        Returns:
            Folder name for saving the image
        """
        target_character = actual_answer
        
        # Validate target character (should be 0-9 for digits, but could be extended for letters)
        if target_character.isdigit():
            digit_num = int(target_character)
            if digit_num < 0 or digit_num > 9:
                raise ValueError("Invalid digit")
            folder_name = f"Digits_{digit_num}"
        else:
            # For future extension to support letters
            folder_name = f"Character_{target_character}"
        
        return folder_name
    
    @staticmethod
    def save_training_image(file_contents: bytes, predicted_digit: str, 
                          actual_answer: str, is_correct: bool) -> Tuple[str, str]:
        """
        Save training image to appropriate folder
        
        Args:
            file_contents: Image file contents
            predicted_digit: What the model predicted
            actual_answer: The correct answer
            is_correct: Whether prediction was correct
            
        Returns:
            Tuple of (file_path, target_character)
        """
        try:
            # Determine the correct folder based on feedback
            is_correct_bool = is_correct
            target_character = actual_answer
            folder_name = TrainingService.determine_target_folder(actual_answer, is_correct_bool)
            
            # Create folder path
            folder_path = os.path.join(settings.TRAINING_DATA_PATH, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}.png"
            file_path = os.path.join(folder_path, filename)
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(file_contents)
            
            print(f"Saved training image: {file_path} (predicted: {predicted_digit}, actual: {actual_answer}, correct: {is_correct})")
            
            return file_path, target_character
            
        except Exception as e:
            print("Error saving training image:", e)
            raise e

# Service instance
training_service = TrainingService()