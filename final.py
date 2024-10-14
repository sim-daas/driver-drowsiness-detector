import cv2
import time
import serial  # For serial communication with Arduino
from ultralytics import YOLO

# Initialize serial communication (adjust 'COM3' or '/dev/ttyUSB0' to your Arduino's port)
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Change to the correct port

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
sleep_duration_threshold = 5  # Threshold in seconds for detecting sleep
sleep_alert_sent = False  # To ensure the signal is sent only once

# Function to check and track sleep state
def check_sleep_state(results):
    global sleep_detected, sleep_start_time, sleep_alert_sent

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
                        if sleep_duration >= sleep_duration_threshold and not sleep_alert_sent:
                            print(f"Sleep detected for more than {sleep_duration_threshold} seconds!")
                            arduino.write(b'1')  # Send signal '1' to Arduino
#                            sleep_alert_sent = True  # Mark the alert as sent to avoid repetition
                else:
                    # Reset if a non-sleep class is detected
                    sleep_detected = False
                    sleep_start_time = None
                    sleep_alert_sent = False  # Reset the alert if sleep is no longer detected

        else:
            # If no detections, reset everything
            sleep_detected = False
            sleep_start_time = None
            sleep_alert_sent = False


# Start capturing and detecting
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO inference on the frame
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

# Release the capture, close serial port and windows
cap.release()
arduino.close()
cv2.destroyAllWindows()
