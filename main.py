from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import logging
from typing import List
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Config ======
MODEL_PATH = "model/ae_user_s036.h5"
SCALER_PATH = "model/scaler_user_s036.pkl"
THRESHOLD = 0.005  # adjust based on your evaluation

# Global variables
model = None
scaler = None

class FeaturesRequest(BaseModel):
    features: List[List[float]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, scaler
    
    logger.info("Loading model and scaler...")
    
    # Check if model directory exists
    if not os.path.exists("model"):
        raise RuntimeError("Model directory 'model' not found")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    
    try:
        # Fix for Keras compatibility issue - load with custom objects
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError(),
            'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
            'MSE': tf.keras.metrics.MeanSquaredError(),
        }
        
        # Try loading with custom objects first
        try:
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            logger.info(f"Model loaded successfully with custom objects from {MODEL_PATH}")
        except Exception as e1:
            logger.warning(f"Failed with custom objects: {e1}")
            # Try loading without compile
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                # Recompile the model manually
                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mse']
                )
                logger.info(f"Model loaded successfully without compile from {MODEL_PATH}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model with both methods: {e1}, {e2}")
                
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Load scaler
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")
    
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")
    
    logger.info("Application startup complete")
    yield
    
    # Shutdown
    logger.info("Application shutdown")

# ====== FastAPI app ======
app = FastAPI(
    title="Keystroke Anomaly Detection API",
    description="API for keystroke dynamics anomaly detection using autoencoder",
    version="1.0.0",
    lifespan=lifespan
)

# ====== Request schema ======
class KeystrokeFeatures(BaseModel):
    features: List[float]  # list of numeric features, same order as training
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        }

# ====== Routes ======
@app.get("/")
def root():
    return {
        "message": "Keystroke Anomaly Detection API is running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold": THRESHOLD
    }

@app.post("/predict")
def predict(data: KeystrokeFeatures):
    """
    Predict if keystroke pattern is anomalous
    """
    try:
        # Validate input
        if not data.features:
            raise HTTPException(status_code=400, detail="Features cannot be empty")
        
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model or scaler not loaded")
        
        # Convert to numpy array
        try:
            X = np.array(data.features, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid features format: {e}")
        
        # Validate feature count
        expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
        if expected_features and X.shape[1] != expected_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {expected_features} features, got {X.shape[1]}"
            )
        
        # Scale features
        try:
            Xs = scaler.transform(X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature scaling failed: {e}")
        
        # Get reconstruction error
        try:
            X_pred = model.predict(Xs, verbose=0)
            rec_error = float(np.mean((X_pred - Xs) ** 2))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
        
        # Decision
        is_anomaly = rec_error > THRESHOLD
        
        return {
            "reconstruction_error": rec_error,
            "threshold": THRESHOLD,
            "is_anomaly": is_anomaly,
            "confidence": abs(rec_error - THRESHOLD) / THRESHOLD if THRESHOLD > 0 else 1.0,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/info")
def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        model_info = {
            "model_type": "Autoencoder",
            "input_shape": list(model.input_shape) if hasattr(model.input_shape, '__iter__') else str(model.input_shape),
            "output_shape": list(model.output_shape) if hasattr(model.output_shape, '__iter__') else str(model.output_shape),
            "threshold": THRESHOLD,
            "scaler_type": type(scaler).__name__ if scaler else None,
            "tensorflow_version": tf.__version__
        }
        
        if hasattr(scaler, 'n_features_in_'):
            model_info["expected_features"] = scaler.n_features_in_
            
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")