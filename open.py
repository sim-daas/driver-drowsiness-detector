import cv2
import numpy as np
from facenet_pytorch import MTCNN

# Step 1: Initialize MTCNN detector
mtcnn = MTCNN(keep_all=True)

# Step 2: Capture video from the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

def is_eye_open(eye_img):
    # Convert to grayscale and apply Gaussian blur to reduce noise
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)

    # Threshold to get dark regions (pupil)
    _, mask = cv2.threshold(blurred_eye, 50, 255, cv2.THRESH_BINARY_INV)
    
    dark_pixels = np.sum(mask == 255)
    total_pixels = eye_img.size // 3
    percentage_dark = (dark_pixels / total_pixels) * 100
    return percentage_dark < 10  # Adjust threshold as necessary

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

    if boxes is not None:
        for i, box in enumerate(boxes):
            # Draw a rectangle around the face
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get eye landmarks
            left_eye = landmarks[i][0]  # (x, y) for left eye
            right_eye = landmarks[i][1]  # (x, y) for right eye

            # Define the bounding box around each eye
            eye_margin = 20  # Margin around the eye for the bounding box
            left_eye_box = [
                max(0, int(left_eye[0] - eye_margin)),
                max(0, int(left_eye[1] - eye_margin)),
                min(frame.shape[1], int(left_eye[0] + eye_margin)),
                min(frame.shape[0], int(left_eye[1] + eye_margin)),
            ]
            right_eye_box = [
                max(0, int(right_eye[0] - eye_margin)),
                max(0, int(right_eye[1] - eye_margin)),
                min(frame.shape[1], int(right_eye[0] + eye_margin)),
                min(frame.shape[0], int(right_eye[1] + eye_margin)),
            ]

            # Get the eye images
            left_eye_img = frame[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]]
            right_eye_img = frame[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]]

            # Check if the eyes are open or closed
            left_eye_state = is_eye_open(left_eye_img)
            right_eye_state = is_eye_open(right_eye_img)

            # Draw the bounding box around the eyes
            left_eye_color = (0, 255, 0) if left_eye_state else (0, 0, 255)
            cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), (left_eye_box[2], left_eye_box[3]), left_eye_color, 2)
            cv2.putText(frame, "Open" if left_eye_state else "Closed",
                        (left_eye_box[0], left_eye_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_eye_color, 2)

            right_eye_color = (0, 255, 0) if right_eye_state else (0, 0, 255)
            cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), (right_eye_box[2], right_eye_box[3]), right_eye_color, 2)
            cv2.putText(frame, "Open" if right_eye_state else "Closed",
                        (right_eye_box[0], right_eye_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_eye_color, 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
