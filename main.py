import io
import json
import logging
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
from pydantic import BaseModel, Field
from typing import List

# --- Pydantic Schemas for Structured Responses ---

class FacePrediction(BaseModel):
    """Schema for a single detected face's prediction."""
    box: List[int] = Field(..., description="Bounding box of the detected face as [x, y, width, height].")
    emotion: str = Field(..., description="Predicted emotion for the face.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the prediction.")

class PredictionResponse(BaseModel):
    """Schema for the final API response."""
    face_count: int = Field(..., description="Total number of faces detected in the image.")
    predictions: List[FacePrediction] = Field(..., description="A list of predictions for each detected face.")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Face Emotion Recognition API",
    description="An API that detects faces in an image and predicts their emotions using a trained Keras model.",
    version="1.0.0"
)

# --- Globals for Models and Labels ---
emotion_model = None
face_cascade = None
labels = None

# --- Startup Event to Load Models ---
@app.on_event("startup")
async def startup_event():
    """
    Load all necessary models and label files on application startup.
    """
    global emotion_model, face_cascade, labels
    try:
        # Load the trained Keras emotion recognition model
        emotion_model = load_model('models/best_model.h5')
        logger.info("✅ Keras emotion model loaded successfully.")

        # Load the pre-trained OpenCV face detector
        cascade_path = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise IOError(f"Unable to load the face cascade classifier from {cascade_path}")
        logger.info("✅ OpenCV face detector loaded successfully.")

        # Load the class labels
        with open('models/labels.json', 'r') as f:
            labels_str_keys = json.load(f)
            labels = {int(k): v for k, v in labels_str_keys.items()}
        logger.info("✅ Class labels loaded successfully.")

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        # Prevent the app from running if essential models fail to load
        emotion_model = None
        face_cascade = None
        labels = None

# --- Helper Function for Image Preprocessing ---
def preprocess_for_emotion_model(face_image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the cropped face image for the emotion recognition model.
    """
    # Convert grayscale face to 3-channel RGB for the model
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    
    # Resize to the model's expected input size
    face_resized = cv2.resize(face_rgb, (128, 128))
    
    # Add a batch dimension
    face_batch = np.expand_dims(face_resized, axis=0)
    
    # Preprocess using EfficientNet's specific function
    # This normalizes the pixels in the way the pre-trained model expects
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(face_batch)

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the Face Emotion Recognition API! Visit /docs for more info."}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Detects faces in an uploaded image and predicts the emotion for each face.
    """
    if not all([emotion_model, face_cascade, labels]):
        raise HTTPException(status_code=503, detail="One or more models are not loaded. Please check server logs.")

    try:
        # Read image file into a format OpenCV can use
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # 1. Detect faces in the image
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return {"face_count": 0, "predictions": []}

        predictions_list = []
        # 2. For each detected face, predict the emotion
        for (x, y, w, h) in faces:
            # Crop the face from the grayscale image
            face_roi = img_gray[y:y+h, x:x+w]

            # Preprocess the face for the emotion model
            processed_face = preprocess_for_emotion_model(face_roi)

            # Predict the emotion
            prediction_probs = emotion_model.predict(processed_face)[0]
            predicted_index = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_index]
            predicted_emotion = labels.get(predicted_index, "Unknown")

            # Store the result
            predictions_list.append(
                FacePrediction(
                    box=[int(x), int(y), int(w), int(h)],
                    emotion=predicted_emotion,
                    confidence=float(confidence)
                )
            )
        
        return {"face_count": len(faces), "predictions": predictions_list}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
