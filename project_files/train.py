import os
import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# Function to compute Haar-like features using OpenCV
def compute_haar_features(image):
    # Define the Haar cascade classifier (you can use pre-trained cascades or train your own)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40))
    
    # Extract Haar-like features
    features = []
    for (x, y, w, h) in faces:
        features.append([x, y, x + w, y + h])  # Rectangular region of detected face
    
    return features

# Directory containing BMP face images for training
train_image_dir = 'training_test_data/training_faces/'

# List all BMP files in the training directory
train_bmp_files = [f for f in os.listdir(train_image_dir) if f.endswith('.bmp')]

# Initialize lists for training features and labels
X_train = []
y_train = []

# Iterate through BMP files in the training directory and extract features
for bmp_file in train_bmp_files:
    image_path = os.path.join(train_image_dir, bmp_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract Haar-like features from the image
    features = compute_haar_features(image)
    
    X_train.extend(features)
    y_train.extend([1] * len(features))  # Assuming all detected regions are faces

# Convert the training feature list to a NumPy array
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

# Train the AdaBoost classifier
ada_boost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
ada_boost_classifier.fit(X_train, y_train)

# Save the trained model to a file using joblib
model_filename = 'face_detection_model.joblib'
joblib.dump(ada_boost_classifier, model_filename)

print(f"Model saved to {model_filename}")
