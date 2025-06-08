"""
Image processing utilities
"""
from PIL import Image, ImageOps
import numpy as np
from app.config.settings import settings

def preprocess_image_for_prediction(img: Image.Image) -> np.ndarray:
    """
    Preprocess image for EMNIST model prediction
    
    Args:
        img: PIL Image object
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    try:
        # Convert to grayscale
        image = img.convert('L')
        
        # Invert to match EMNIST (white foreground on black background)
        image = ImageOps.invert(image)
        
        # Crop to bounding box (removes blank space)
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        
        # Pad and resize to 28x28 (centering content)
        image = ImageOps.pad(image, (28, 28), color=0, centering=(0.5, 0.5))
        
        # EMNIST-specific orientation correction (commented out for now)
        # image = image.rotate(270, expand=False)  # Rotate 270° clockwise
        # image = ImageOps.mirror(image)           # Flip left-right
        
        # Normalize and reshape for model
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # (28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)   # (1, 28, 28, 1)
        
        # Print debug info
        print(f"Processed image shape: {img_array.shape}")
        print(f"Image min/max: {img_array.min():.3f}/{img_array.max():.3f}")
        
        return img_array
        
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        raise e