import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from playsound import playsound
import threading
import time


model = load_model('driver_drowsiness_vgg16.h5')

cap = cv2.VideoCapture(0)

IMG_SIZE = 227  # Must match your model input size
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def preprocess_frame_with_face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No face detected

    # Take the first detected face (you can enhance for multiple faces if needed)
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

def alert_drowsiness():
    threading.Thread(target=playsound, args=('alarm.wav',), daemon=True).start()

last_alert_time = 0
alert_cooldown = 5
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input
    input_data = preprocess_frame_with_face_detection(frame)
    prediction = model.predict(input_data)

    # Classify
    label = "Drowsy" if prediction[0][0] > 0.7 else "Alert"
    color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

    # Show result
    cv2.putText(frame, f'Status: {label}', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Driver Drowsiness Detection', frame)
   
    prediction = model.predict(input_data)
    print("Prediction raw output:", prediction)

    if label == "Drowsy":
        current_time = time.time()
        if current_time - last_alert_time > alert_cooldown:
            print("ALERT: Driver is Drowsy!")
            alert_drowsiness()
            last_alert_time = current_time
    else:
        print("Driver is not drowsy.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

# img = cv2.imread("Drowsy.png")
# img_resized = preprocess_frame_with_face_detection(img)
# prediction = model.predict(img_resized)
# print(prediction)
# label = "Drowsy" if prediction[0][0] > 0.5 else "Alert"
# print(label)
