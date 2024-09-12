import os
import cv2
import tensorflow as tf
from mtcnn import MTCNN

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize MTCNN face detector
detector = MTCNN()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(rgb_frame)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result in a window
    cv2.imshow('Webcam - MTCNN Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
