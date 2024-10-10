import cv2
import os

# Function to brighten an image
def brighten_image(image_path, alpha=1.0, beta=50):
    image = cv2.imread(image_path)
    bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return bright_image

# Input and output folders
input_folder = 'eyes/open'     # Folder containing original images
output_folder = 'eyes/bopen'  # Folder to save the brightened images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg')):
        input_image_path = os.path.join(input_folder, filename)
        
        # Brighten the image
        bright_image = brighten_image(input_image_path, alpha=1.0, beta=50)

        # Save the brightened image in the output folder
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, bright_image)

print("All images have been brightened and saved in the output folder.")
