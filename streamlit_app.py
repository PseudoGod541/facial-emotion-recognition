import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FastAPI Backend URL ---
# For local testing, this is typically http://127.0.0.1:8000/predict/
# If using Docker Compose, change '127.0.0.1' to the service name (e.g., 'api')
API_URL = "http://api:8000/predict/"

# --- Helper Function to Draw on Image ---
def draw_predictions_on_image(image: Image.Image, predictions: list) -> Image.Image:
    """
    Draws bounding boxes and emotion labels on the image.
    """
    # Convert PIL Image to an OpenCV image (numpy array)
    img_cv = np.array(image.convert('RGB'))
    
    # Define colors for emotions (optional, for better visualization)
    emotion_colors = {
        "happy": (0, 255, 0),      # Green
        "sad": (255, 0, 0),        # Blue
        "angry": (0, 0, 255),      # Red
        "surprise": (0, 255, 255), # Cyan
        "neutral": (200, 200, 200),# Gray
        "fear": (128, 0, 128),     # Purple
        "disgust": (0, 128, 0),    # Dark Green
    }
    default_color = (255, 255, 255) # White

    for pred in predictions:
        box = pred['box']
        emotion = pred['emotion']
        confidence = pred['confidence']
        
        x, y, w, h = box
        color = emotion_colors.get(emotion.lower(), default_color)
        
        # Draw the bounding box
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
        
        # Create the label text
        label = f"{emotion}: {confidence:.2f}"
        
        # Put the label above the bounding box
        cv2.putText(img_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    # Convert the OpenCV image back to a PIL Image
    return Image.fromarray(img_cv)

# --- UI Design ---
st.title("😊 Face Emotion Recognition")
st.write(
    "Upload an image with faces, and the model will detect the faces and "
    "predict the emotion for each one."
)

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # --- Prediction Logic ---
    if st.button('Recognize Emotions', key='predict_button'):
        with st.spinner('Sending image to the model for analysis...'):
            try:
                # Prepare the file for the POST request
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send the POST request
                response = requests.post(API_URL, files=files)
                response.raise_for_status()  # Raise an exception for bad status codes

                result = response.json()
                face_count = result.get('face_count', 0)
                predictions = result.get('predictions', [])

                st.subheader("🧠 Analysis Result")
                if face_count > 0:
                    st.success(f"Detected {face_count} face(s) in the image.")
                    
                    # Draw predictions on the image and display it
                    annotated_image = draw_predictions_on_image(image, predictions)
                    st.image(annotated_image, caption='Image with Emotion Predictions', use_column_width=True)
                    
                    # Display detailed results in an expander
                    with st.expander("See detailed results"):
                        st.table(predictions)
                else:
                    st.warning("No faces were detected in the image.")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please ensure the backend is running. Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
