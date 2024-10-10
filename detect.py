import cv2
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms, models
from PIL import Image

# Step 1: Load the fine-tuned ResNet-18 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Two output classes: open/closed
model.load_state_dict(torch.load('eye_classifier_resnet18.pth'))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Step 2: Initialize MTCNN detector for face and eye detection
mtcnn = MTCNN(keep_all=True, device=device)

# Step 3: Define transforms for the eye images (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Step 4: Function to classify eye state (open or closed)
def classify_eye(eye_img):
    eye_img = transform(eye_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(eye_img)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()  # 0 = Closed, 1 = Open

# Step 5: Capture video from camera
cap = cv2.VideoCapture(0)  # 0 is the default camera
frame_width = 1280  # Desired width
frame_height = 720  # Desired height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB (for MTCNN)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and facial landmarks
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    # If faces are detected
    if boxes is not None:
        for i, box in enumerate(boxes):
            # Draw a rectangle around the face
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get eye landmarks
            left_eye = landmarks[i][0]  # (x, y) for left eye
            right_eye = landmarks[i][1]  # (x, y) for right eye

            # Define the bounding box around each eye
            eye_margin = 10  # Margin around the eye for the bounding box
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

            # Extract eye regions from the frame
            left_eye_img = Image.fromarray(frame[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]])
            right_eye_img = Image.fromarray(frame[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]])

            # Classify left and right eyes
            left_eye_state = classify_eye(left_eye_img)
            right_eye_state = classify_eye(right_eye_img)

            # Draw rectangles around the eyes and show open/closed state
            eye_color = (0, 0, 255) if left_eye_state == 0 else (0, 255, 0)  # Red for closed, green for open
            cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), (left_eye_box[2], left_eye_box[3]), eye_color, 2)
            cv2.putText(frame, "Closed" if left_eye_state == 0 else "Open", 
                        (left_eye_box[0], left_eye_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)

            eye_color = (0, 0, 255) if right_eye_state == 0 else (0, 255, 0)
            cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), (right_eye_box[2], right_eye_box[3]), eye_color, 2)
            cv2.putText(frame, "Closed" if right_eye_state == 0 else "Open", 
                        (right_eye_box[0], right_eye_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)

    # Display the video with detected faces and eye state
    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
