import os
import cv2
import joblib

# Load the trained model
model_filename = 'face_detection_model.joblib'
ada_boost_classifier = joblib.load(model_filename)

# Function to compute Haar-like features using OpenCV
def compute_haar_features(image):
    # Define the Haar cascade classifier (you can use pre-trained cascades or train your own)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Extract Haar-like features
    features = []
    for (x, y, w, h) in faces:
        features.append([x, y, x + w, y + h])  # Rectangular region of detected face
    
    return features

# Directory containing JPG images for face detection
test_image_dir = 'training_test_data/test_face_photos/'

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the loaded AdaBoost classifier
    faces = compute_haar_features(gray)
    
    # Draw rectangles around detected faces
    for (x, y, x2, y2) in faces:
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    
    # Save the processed image in the output directory
    output_path = os.path.join(output_dir, jpg_file)
    cv2.imwrite(output_path, image)

print("Processed images saved to the 'output' directory.")
