'''
import cv2
import time
from ultralytics import YOLO

# Load the YOLOv11 model (Replace 'better.pt' with your model's file)
model = YOLO("better.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to track sleep state
sleep_detected = False
sleep_start_time = None
sleep_duration_threshold = 5  # Time (in seconds) to detect prolonged sleep
sleep_alert_printed = False  # To ensure "Sleep" is printed only once

# Set confidence threshold
confidence_threshold = 0.6  # Adjust based on your requirement

# Function to check and track sleep state
def check_sleep_state(results):
    global sleep_detected, sleep_start_time, sleep_alert_printed

    # Iterate over results to check for "sleep" class (assuming it's class 0)
    for result in results:
        boxes = result.boxes

        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                cls = int(boxes.cls[i])  # Get class label
                conf = float(boxes.conf[i].item())
#                print(conf)
                # Check if the detection is "closed eyes/sleep" (class 0) and the confidence is above the threshold
                if cls == 0 and conf >= confidence_threshold:
                    if not sleep_detected:
                        sleep_detected = True
                        sleep_start_time = time.time()  # Record the time when sleep is first detected
                    else:
                        # Calculate how long sleep has been detected
                        sleep_duration = time.time() - sleep_start_time
                        if sleep_duration >= sleep_duration_threshold and not sleep_alert_printed:
                            print(f"Sleep detected for more than {sleep_duration_threshold} seconds!")
#                            sleep_alert_printed = True  # Mark the alert as printed to avoid repetition

                # If a non-sleep class is detected or confidence is below threshold, reset the state
                else:
                    sleep_detected = False
                    sleep_start_time = None
                    sleep_alert_printed = False

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

    # Draw YOLO results on the frame
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                conf = boxes.conf[i]  # Confidence score
                cls = int(boxes.cls[i])  # Class label

                # Draw the bounding box and label on the frame
                if conf >= confidence_threshold:  # Only display results above confidence threshold
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
import serial

# Load the YOLOv11 model (Replace 'model.pt' with your model's file)
model = YOLO("better.pt")
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Change to the correct p>

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to track sleep state
sleep_detected = False
sleep_start_time = None
sleep_duration_threshold = 2
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
                conf = float(boxes.conf[i].item())
                # Assuming class 0 is "closed eyes/sleep" (adjust based on your model's labels)
                if cls == 0 and conf >= 0.6:
                    if not sleep_detected:
                        sleep_detected = True
                        sleep_start_time = time.time()  # Record the time when sleep is first detected
                    else:
                        # Calculate how long sleep has been detected
                        sleep_duration = time.time() - sleep_start_time
                        if sleep_duration >= sleep_duration_threshold and not sleep_alert_printed:
                            print(sleep_start_time, time.time(), sleep_duration)
                            print("Sleep detected for more than 2 seconds!")
                            arduino.write(b'1')
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

