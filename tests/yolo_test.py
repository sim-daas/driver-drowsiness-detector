import cv2
import torch
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8m.pt")

# Set to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load class names from labels.txt
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (YOLOv8 expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model.predict(source=rgb_frame, stream=True)

    for result in results:
        # Convert results to numpy arrays
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Draw bounding boxes and labels on the frame
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[int(cls_id)]}: {conf:.2f}" if int(cls_id) < len(class_names) else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
