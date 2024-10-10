'''
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")  # Replace 'model.pt' with your model path

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change 0 if using another camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to perform inference with streaming and handle detections
def stream_inference(model, cap):
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run inference with stream=True (memory-efficient)
        results = model.predict(source=frame, stream=True)

        # Iterate through results (each result corresponds to an image/frame)
        for result in results:
            # Access detection details
            boxes = result.boxes  # Bounding boxes
            masks = result.masks  # Segmentation masks (if present)
            probs = result.probs  # Classification probabilities (if present)

            # Drawing bounding boxes on the frame
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)  # Convert to int for OpenCV
                    conf = boxes.conf[i]  # Confidence score
                    cls = boxes.cls[i]  # Class label

                    # Draw rectangle and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{cls} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If segmentation exists, process it (optional)
            if masks is not None:
                for mask in masks.data:
                    # Draw the segmentation mask (use alpha blending or contours if needed)
                    pass  # Add segmentation processing code here

            # Display the frame with annotations
            cv2.imshow('YOLOv11 Stream Inference', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start streaming inference
stream_inference(model, cap)

# When done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
'''

import cv2
import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv11 model
model = YOLO("best.pt")  # Replace 'model.pt' with your YOLOv11 model path

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change 0 if using another camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to perform inference on face regions
def face_region_inference(model, frame, faces):
    for face_box in faces:
        x1, y1, x2, y2 = map(int, face_box)  # Get bounding box coordinates
        face_region = frame[y1:y2, x1:x2]  # Extract face region

        # Run YOLOv11 inference on the face region
        results = model.predict(source=face_region)

        # Draw YOLO results on the face region
        if results:
            result = results[0]  # Get the first (and likely only) result
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    fx1, fy1, fx2, fy2 = map(int, box)
                    conf = boxes.conf[i]  # Confidence score
                    cls = boxes.cls[i]    # Class label

                    # Adjust the coordinates to the original image
                    cv2.rectangle(frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (0, 255, 0), 2)
                    label = f'{cls} {conf:.2f}'
                    cv2.putText(frame, label, (x1 + fx1, y1 + fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Start capturing and detecting
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        # Draw face bounding boxes detected by MTCNN
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle for MTCNN face detection

        # Perform YOLO inference on the detected face regions
        frame = face_region_inference(model, frame, boxes)

    # Show the frame with both face detection and YOLO annotations
    cv2.imshow('Face and YOLO Inference', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()

