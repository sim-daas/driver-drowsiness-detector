import cv2
import torch
import os
from facenet_pytorch import MTCNN
import time

# Step 3: Initialize the MTCNN detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)  # keep_all=True to detect multiple faces

# Step 4: Create a directory to save eye images
output_dir = 'eyes/closed'
os.makedirs(output_dir, exist_ok=True)

# Step 5: Function to perform face and eye detection and save eye images
def detect_faces_and_save_eyes(frame, frame_count):
    # Convert the frame from BGR (OpenCV format) to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and facial landmarks using MTCNN
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    # If no faces are detected, return the frame as is
    if boxes is None:
        return frame

    # Loop through detected faces
    for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
        # Draw a rectangle around the face
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract eye positions from the landmarks
        left_eye = landmark[0]  # Left eye (x, y)
        right_eye = landmark[1]  # Right eye (x, y)

        # Define the square size (e.g., 20x20 pixels)
        square_size = 20
        i = 5
        # Crop and save the left and right eye regions
        for eye_name, eye in zip(['left', 'right'], [left_eye, right_eye]):
            eye_x, eye_y = eye.astype(int)
            eye_crop = frame[eye_y-square_size//2:eye_y+square_size//2, eye_x-square_size//2:eye_x+square_size//2]

            # Save the cropped eye image
            eye_filename = os.path.join(output_dir, f'frame_{frame_count}_face_{i}_{eye_name}_eye.jpg')
            cv2.imwrite(eye_filename, eye_crop)

            # Draw square bounding boxes around the eyes on the frame
            cv2.rectangle(frame, (eye_x - square_size//2, eye_y - square_size//2),
                          (eye_x + square_size//2, eye_y + square_size//2), (255, 0, 0), 2)

    return frame

# Step 6: Function to capture video from webcam or file and detect eyes
def process_video(input_source=0):  # 0 for webcam, or provide a video file path
    # Capture video from webcam or file
    cap = cv2.VideoCapture(input_source)
    frame_width = 1280  # Desired width
    frame_height = 720  # Desired height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    frame_count = 0  # To keep track of frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection and save eye images
        processed_frame = detect_faces_and_save_eyes(frame, frame_count)

        # Display the frame with detections
        cv2.imshow('Drowsiness Detection', processed_frame)

        # Increment frame count
        frame_count += 1

        # Exit the video loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Step 7: Run the video processing (0 for webcam, or provide video file path)
process_video(0)  # Use `process_video('video.mp4')` for video file input
