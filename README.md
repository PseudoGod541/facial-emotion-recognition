# Real-Time Face Emotion Recognition Application

This is a full-stack computer vision project that detects faces in an uploaded image and classifies the emotion of each detected face. The system uses a deep learning model served by a FastAPI backend, with an interactive Streamlit frontend for visualization. The entire application is containerized for easy deployment with Docker.

![Streamlit Frontend Screenshot](<path_to_your_screenshot.png>) <!-- Add a screenshot of your Streamlit app here -->

---

## 📋 Features

-   **Deep Learning Model**: A powerful model trained on the FER2013 dataset using transfer learning with `EfficientNetB0`.
-   **Face Detection**: Utilizes OpenCV's Haar Cascade classifier to first locate faces within the image.
-   **FastAPI Backend**: A robust REST API that processes an image, detects faces, classifies emotions, and returns bounding box coordinates and predictions.
-   **Streamlit Frontend**: An interactive web app that allows users to upload an image and see the detected emotions drawn directly onto the faces.
-   **Dockerized**: Fully containerized using Docker and Docker Compose for a seamless, one-command setup.

---

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Machine Learning**: TensorFlow, Keras, OpenCV, Pillow, NumPy
-   **Frontend**: Streamlit, Requests
-   **Deployment**: Docker, Docker Compose

---

## 🚀 How to Run

To run this application, you need to have Docker and Docker Compose installed.
```bash
1. Clone the Repository


git clone <https://github.com/PseudoGod541/facial-emotion-recognition/>
cd <your-project-directory>

2. Place Required Files
Ensure the following files are in your project directory:

A models/ folder containing best_model.h5 and labels.json.

The OpenCV face detector file: haarcascade_frontalface_default.xml.

3. Run with Docker Compose
This single command will build the Docker image and start both the API and frontend services.

docker-compose up --build

4. Access the Application
Streamlit Frontend: Open your browser and go to http://localhost:8501

FastAPI Backend Docs: Open your browser and go to http://localhost:8000/docs

📁 Project Structure
.
├── models/
│   ├── best_model.h5
│   └── labels.json
├── main.py                               # FastAPI application
├── streamlit_app.py                      # Streamlit frontend application
├── haarcascade_frontalface_default.xml   # OpenCV face detector
├── Dockerfile                            # Instructions to build the Docker image
├── docker-compose.yml                    # Defines and runs the multi-container setup
├── .dockerignore                         # Specifies files to ignore during build
└── requirements.txt                      # Python dependencies
