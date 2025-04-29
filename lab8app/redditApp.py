from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import mlflow
import os
import numpy as np
import warnings
from typing import Dict, Any

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define our FastAPI app
app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Define the input data model
class RedditComment(BaseModel):
    reddit_comment: str

# Define the prediction response model
class PredictionResponse(BaseModel):
    Predictions: Dict[str, float]

# Global variable to store the loaded model
MODEL = None

# Load the model
def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    
    try:
        # Try loading from MLflow
        try:
            print("Attempting to load model from MLflow")
            mlflow_model_path = "models:/reddit_classifier/latest"
            MODEL = mlflow.pyfunc.load_model(mlflow_model_path)
            print("Successfully loaded model from MLflow")
        except Exception as e:
            print(f"Failed to load from MLflow: {e}")
            # Fallback to local joblib file
            print("Attempting to load model from local file")
            MODEL = joblib.load("/Users/bensunshine/repos/msds603/reddit_model_pipeline.joblib")
            print("Successfully loaded model from local file")
        
        return MODEL
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise Exception("Failed to load model from both MLflow and local file.")

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying Reddit comments'}

# Prediction endpoint
@app.post('/predict', response_model=PredictionResponse)
def predict(comment: RedditComment):
    try:
        model = load_model()
        text = [comment.reddit_comment]
        
        print(f"Making prediction for: {text}")
        
        # Get prediction probabilities
        predictions = model.predict_proba(text)
        
        # Format results - probability of being removed (class 1)
        prob_remove = float(predictions[0][1])
        prob_keep = float(predictions[0][0])
        
        print(f"Prediction result: {prob_keep} (keep), {prob_remove} (remove)")
        
        # Return the prediction in the required format
        return {"Predictions": {str(prob_keep): prob_remove}}
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Load the model when the application starts
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("Model loaded successfully during startup")
    except Exception as e:
        print(f"Warning: Failed to preload model during startup: {e}")
        # We don't raise an exception here to allow the app to start
        # The model will be loaded on the first prediction request

if __name__ == "__main__":
    uvicorn.run("redditApp:app", host="0.0.0.0", port=8000, reload=True) 