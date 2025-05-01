import streamlit as st
import cv2
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
import tensorflow as tf
from playsound import playsound
import threading

# Model URL and Path
MODEL_URL = "https://drive.google.com/uc?export=download&id=1XDcuo7AmPKPlUmS83IWhOZ5y3cBWHbHt"
MODEL_PATH = "driver_drowsiness_vgg16.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.write("Download complete.")

# Download the model (only if it's not already present)
download_model()

# Load the model
model = load_model(MODEL_PATH)

# Image pre-processing function
IMG_SIZE = 227  # Model input size
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def preprocess_frame_with_face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No face detected

    (x, y, w, h) = faces[0]
    face_img = frame[y:y+h, x:x+w]

    # Resize to model input size
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0

    # Convert to tensor with batch dimension
    input_tensor = tf.convert_to_tensor(face_normalized, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    return input_tensor

# Alert function
def alert_drowsiness():
    threading.Thread(target=playsound, args=('alarm.wav',), daemon=True).start()

# Streamlit UI
st.title("Driver Drowsiness Detection")
st.write("This app detects whether a driver is drowsy or not based on their face and eye movements.")

# Start video capture
cap = cv2.VideoCapture(0)

# Loop for real-time webcam detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and make predictions
    input_data = preprocess_frame_with_face_detection(frame)
    
    if input_data is not None:
        prediction = model.predict(input_data)
        label = "Drowsy" if prediction[0][0] > 0.7 else "Alert"
        
        # Display result on frame
        color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
        cv2.putText(frame, f'Status: {label}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show frame in Streamlit (update the image, not stack it)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Play alert sound if drowsy
        if label == "Drowsy":
            alert_drowsiness()
            st.write("ALERT: Driver is Drowsy!")
        else:
            st.write("Driver is Alert!")

    # Wait for 'q' to exit webcam streaming loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
