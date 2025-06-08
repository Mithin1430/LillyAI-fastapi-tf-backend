#!/usr/bin/env python3
"""
Startup script for LillyAI Digit Recognition API
"""
import uvicorn
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # Change to the API directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    MODEL_PATH = os.getenv("MODEL_PATH", "models/AlphaNumeric/model.h5")
    
    print("Starting LillyAI Digit Recognition API...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Debug mode: {DEBUG}")
    print(f"Checking for model file: {MODEL_PATH}")
    
    if os.path.exists(MODEL_PATH):
        print("✓ Model file found!")
    else:
        print("✗ Model file not found!")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug"
    )
