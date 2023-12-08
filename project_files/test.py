# Import configuration variables
from config import data_directory, training_directory, code_directory

import os
import cv2
import numpy as np
import joblib
import matplotlib
matplotlib.use('TkAgg')  # Adjust as needed for your environment

# Custom Decision Stump Classifier
class DecisionStump:
    def __init__(self):
        self.best_feature = None
        self.best_threshold = None
        self.best_rule = None

    def predict(self, X):
        feature_values = X[:, self.best_feature]
        return np.where(feature_values * self.best_rule < self.best_threshold * self.best_rule, 1, -1)

def detect_skin_ycbcr(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input")

    LOWER_SKIN = np.array([0, 133, 50], dtype=np.uint8)
    UPPER_SKIN = np.array([250, 180, 135], dtype=np.uint8)

    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycbcr_image, LOWER_SKIN, UPPER_SKIN)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin, skin_mask

def compute_haar_features(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    features = []
    for (x, y, w, h) in faces:
        features.append([x, y, x + w, y + h])
    return features

def adaboost_predict(classifiers, alpha_values, X):
    final_predictions = [sum(alpha * clf.predict(np.array([x])) for alpha, clf in zip(alpha_values, classifiers)) for x in X]
    return [1 if prediction > 0 else -1 for prediction in final_predictions]

# Assuming 'data' directory is at the same level as 'training_test_data'
model_directory = os.path.join(os.path.dirname(data_directory), 'data')
model_filename = os.path.join(model_directory, 'face_detection_model.joblib')

output_dir = os.path.join(code_directory, 'output')
os.makedirs(output_dir, exist_ok=True)

# Paths to test images are directly inside 'training_test_data'
test_image_dirs = ['test_cropped_faces', 'test_face_photos', 'test_nonfaces']

# Output directory (relative path)
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Load AdaBoost classifier
ada_boost_classifier = joblib.load(model_filename)

# Processing loop
for dir in test_image_dirs:
    current_test_dir = os.path.join(training_directory, dir)
    test_jpg_files = [f for f in os.listdir(current_test_dir) if f.endswith('.jpg') or f.endswith('JPG')]

    for jpg_file in test_jpg_files:
        image_path = os.path.join(current_test_dir, jpg_file)
        image = cv2.imread(image_path)

        skin_detected, skin_mask = detect_skin_ycbcr(image)
        gray_skin = cv2.cvtColor(skin_detected, cv2.COLOR_BGR2GRAY)
        faces = compute_haar_features(gray_skin)

        X_test = np.array(faces, dtype=np.float32)
        predictions = adaboost_predict(ada_boost_classifier[0], ada_boost_classifier[1], X_test)

        for face, prediction in zip(faces, predictions):
            if prediction == 1:
                x, y, x2, y2 = face
                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

        output_path = os.path.join(output_dir, os.path.basename(dir), jpg_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

print("Processed images saved to the 'output' directory.")
