from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "models/AlphaNumeric/model.h5")
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "Digits")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Initialize FastAPI app
app = FastAPI(
    title="LillyAI Character Recognition API",
    description="AI-powered character recognition for educational games using EMNIST",
    version="2.0.0",
    debug=DEBUG
)

# Allow CORS (for frontend usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EMNIST ByClass label mapping (0–61)
emnist_byclass_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

# Load Keras model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"EMNIST model loaded successfully from: {MODEL_PATH}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Model supports {len(emnist_byclass_labels)} character classes")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Make sure you have the model.h5 file in the modals/AlphaNumeric/ directory")
    print(f"Expected path: {os.path.abspath(MODEL_PATH)}")
    model = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "LillyAI Character Recognition API", 
        "status": "running",
        "environment": ENVIRONMENT,
        "version": "2.0.0",
        "model_type": "EMNIST ByClass",
        "supported_characters": len(emnist_byclass_labels)
    }

@app.get("/health")
async def health():
    model_status = "loaded" if model is not None else "failed"
    model_info = {}
    if model is not None:
        model_info = {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "model_path": MODEL_PATH,
            "model_type": "EMNIST ByClass",
            "supported_characters": len(emnist_byclass_labels),
            "character_classes": emnist_byclass_labels[:10] + ["..."] + emnist_byclass_labels[-10:]
        }
    return {
        "status": "healthy", 
        "model": model_status,
        "model_info": model_info,
        "config": {
            "host": HOST,
            "port": PORT,
            "debug": DEBUG,
            "training_data_path": TRAINING_DATA_PATH,
            "environment": ENVIRONMENT
        }
    }

# Function to preprocess and predict using EMNIST model
def predict_character_from_image(img: Image.Image):
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
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        label_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        predicted_char = emnist_byclass_labels[label_index]
        
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Top 5 predictions: {np.argsort(predictions[0])[-5:][::-1]}")
        print(f"Predicted character: '{predicted_char}' (index: {label_index}), Confidence: {confidence:.3f}")
        
        return label_index, predicted_char, confidence
        
    except Exception as e:
        print(f"Error in character prediction: {e}")
        raise e

# Endpoint to handle file upload and prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if model is loaded
        if model is None:
            return JSONResponse(
                content={"error": "EMNIST model not loaded. Please ensure model.h5 exists."}, 
                status_code=500
            )
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "Invalid file type. Please upload an image."}, 
                status_code=400
            )
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        label_index, predicted_char, confidence = predict_character_from_image(img)

        # For backward compatibility with frontend expecting "prediction" field
        # If it's a digit (0-9), return the numeric value, otherwise return the character
        if predicted_char.isdigit():
            prediction_value = int(predicted_char)
        else:
            prediction_value = predicted_char

        return JSONResponse(content={
            "prediction": prediction_value,
            "label": label_index,
            "character": predicted_char,
            "confidence": confidence,
            "status": "success"
        })

    except Exception as e:
        print("Prediction error:", e)
        return JSONResponse(
            content={"error": str(e), "status": "error"}, 
            status_code=500
        )

# Endpoint to save training images
@app.post("/save_training_image")
async def save_training_image(
    file: UploadFile = File(...),
    predicted_digit: str = Form(...),
    actual_answer: str = Form(...),
    is_correct: str = Form(...)
):
    try:
        # Determine the correct folder based on feedback
        is_correct_bool = is_correct.lower() == 'true'
        target_character = actual_answer if is_correct_bool else actual_answer
        
        # Validate target character (should be 0-9 for digits, but could be extended for letters)
        if target_character.isdigit():
            digit_num = int(target_character)
            if digit_num < 0 or digit_num > 9:
                raise ValueError("Invalid digit")
            folder_name = f"Digits_{digit_num}"
        else:
            # For future extension to support letters
            folder_name = f"Character_{target_character}"
        
        # Create folder path
        folder_path = os.path.join(TRAINING_DATA_PATH, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}.png"
        file_path = os.path.join(folder_path, filename)
        
        # Save the file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        print(f"Saved training image: {file_path} (predicted: {predicted_digit}, actual: {actual_answer}, correct: {is_correct})")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Training image saved successfully",
            "file_path": file_path,
            "character": target_character
        })
        
    except Exception as e:
        print("Error saving training image:", e)
        return JSONResponse(
            content={"error": str(e), "status": "error"}, 
            status_code=500
        )
