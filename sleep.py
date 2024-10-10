'''
import cv2
import torch
import time
from ultralytics import YOLO

# Load the YOLOv11 model (Replace 'model.pt' with your model's file)
model = YOLO("best.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to track sleep state
sleep_detected = False
sleep_start_time = None
sleep_duration_threshold = 20
z = 0
# Function to check and track sleep state
def check_sleep_state(results):
    global sleep_detected, sleep_start_time, z
    
    # Iterate over results to check for "sleep" class (assuming it's class 1, adjust as per your model)
    for result in results:
        boxes = result.boxes
        
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                cls = int(boxes.cls[i])  # Get class label
                
                # Assuming class 1 is "sleep" (adjust this based on your model's class labels)
                if cls == 0:  # 1 corresponds to "sleep" in this example
                    if not sleep_detected:
                        sleep_detected = True
                        sleep_start_time = time.time()  # Record the time when sleep is first detected
                    else:
                        # Calculate how long sleep has been detected
                        sleep_duration = time.time() - sleep_start_time
                        if sleep_duration >= sleep_duration_threshold:
                            print("Sleep detected for more than 4 seconds!", z)
                            z += 1
                else:
                    # Reset if the "sleep" class is not detected
                    sleep_detected = False
                    sleep_start_time = None

# Start capturing and detecting
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv11 inference on the frame
    results = model.predict(source=frame, verbose=False)

    # Check if "sleep" state is detected and track its duration
    check_sleep_state(results)

    # Draw YOLO results on the frame
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                conf = boxes.conf[i]  # Confidence score
                cls = int(boxes.cls[i])  # Class label

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Class {cls}, Conf: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Inference with Sleep Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
'''


import cv2
import time
from ultralytics import YOLO

# Load the YOLOv11 model (Replace 'model.pt' with your model's file)
model = YOLO("better.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to track sleep state
sleep_detected = False
sleep_start_time = None
sleep_duration_threshold = 5
sleep_alert_printed = False  # To ensure "Sleep" is printed only once

# Function to check and track sleep state
def check_sleep_state(results):
    global sleep_detected, sleep_start_time, sleep_alert_printed

    # Iterate over results to check for "sleep" class (assuming it's class 1, adjust as per your model)
    for result in results:
        boxes = result.boxes

        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                cls = int(boxes.cls[i])  # Get class label

                # Assuming class 0 is "closed eyes/sleep" (adjust based on your model's labels)
                if cls == 0:
                    if not sleep_detected:
                        sleep_detected = True
                        sleep_start_time = time.time()  # Record the time when sleep is first detected
                    else:
                        # Calculate how long sleep has been detected
                        sleep_duration = time.time() - sleep_start_time
                        if sleep_duration >= sleep_duration_threshold and not sleep_alert_printed:
                            print(sleep_start_time, time.time(), sleep_duration)
                            print("Sleep detected for more than 2 seconds!")
                            sleep_detected = False
                            sleep_start_time = None
                            sleep_alert_printed = False
#                            sleep_alert_printed = True  # Mark the alert as printed to avoid repetition
                else:
                    # If a non-sleep class is detected, we should reset the state
                    sleep_detected = False
                    sleep_start_time = None
                    sleep_alert_printed = False  # Reset the alert state if sleep is no longer detected

        else:
            sleep_detected = False
            sleep_start_time = None
            sleep_alert_printed = False


# Start capturing and detecting
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv11 inference on the frame
    results = model.predict(source=frame, verbose=False)

    # Check if "sleep" state is detected and track its duration
    check_sleep_state(results)
#    print(results)

    # Draw YOLO results on the frame
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                conf = boxes.conf[i]  # Confidence score
                cls = int(boxes.cls[i])  # Class label

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Class {cls}, Conf: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Inference with Sleep Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

