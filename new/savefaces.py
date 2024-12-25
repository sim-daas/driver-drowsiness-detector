import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import numpy as np
import time
# Load the YOLOv11 model
model = YOLO("models/yolov11n-face.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the minimum confidence threshold for detection
min_confidence = 0.5

# Create a directory to store the extracted faces
extract_folder = "faces"
os.makedirs(extract_folder, exist_ok=True)

count = 0

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    
    if not success:
        break
    
    # Convert the frame to RGB and preprocess it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, conf=min_confidence)
    
    # Draw bounding boxes and labels for detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), \
                            int(box.xyxy[0][2]), int(box.xyxy[0][3])
            
            # Check if the object is a face (assuming faces are class 0)
            if box.cls[0] == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Get the confidence value from the tensor
                confidence = round(float(box.conf[0].item()), 2)
                label = f"{model.names[int(box.cls[0])]} {confidence}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                # Extract the face
                face = frame[y1:y2, x1:x2]
                
                # Save the extracted face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{extract_folder}/face_{count}_{timestamp}.jpg"
                cv2.imwrite(filename, face)
                
                count += 1
    
    # Display the frame with detections
    cv2.imshow("Face Detection", frame)
    time.sleep(0.1)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Extracted {count} faces and saved them to {extract_folder}")
