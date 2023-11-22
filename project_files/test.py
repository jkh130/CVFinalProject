import os
import cv2
import numpy as np
import joblib


def detect_skin_ycbcr(image):
    # Convert the image to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Define the range of skin color in YCbCr
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Threshold the YCbCr image to get only skin colors
    skin_mask = cv2.inRange(ycbcr_image, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin, skin_mask

def compute_haar_features(image):
    # Define the Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract Haar-like features
    features = []
    for (x, y, w, h) in faces:
        features.append([x, y, x + w, y + h])

    return features


# Load the trained model
model_filename = 'face_detection_model.joblib'
ada_boost_classifier = joblib.load(model_filename)

# Directory containing JPG images for face detection
test_image_dir = '../training_test_data/test_face_photos/'

# Output directory to save processed images with rectangles
output_dir = 'output/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all JPG files in the test directory
test_jpg_files = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg') or f.endswith('JPG')]

# Iterate through JPG files in the test directory and apply face detection
for jpg_file in test_jpg_files:
    image_path = os.path.join(test_image_dir, jpg_file)
    image = cv2.imread(image_path)

    # Apply skin detection
    skin_detected, skin_mask = detect_skin_ycbcr(image)

    # Convert skin-detected image to grayscale for Haar feature computation
    gray_skin = cv2.cvtColor(skin_detected, cv2.COLOR_BGR2GRAY)

    # Detect faces using the loaded AdaBoost classifier on the skin-filtered image
    faces = compute_haar_features(gray_skin)

    # Draw rectangles around detected faces on the original image
    for (x, y, x2, y2) in faces:
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

    # Save the processed image in the output directory
    output_path = os.path.join(output_dir, jpg_file)
    cv2.imwrite(output_path, image)

print("Processed images saved to the 'output' directory.")
