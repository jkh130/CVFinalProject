import os
import cv2
import numpy as np
import pickle
from sklearn.metrics import classification_report

# Constants
IMAGE_SIZE = (24, 24)  # Standardizing images to 24x24 pixels
DATA_DIR = 'training_test_data'  # Base directory for the data

def load_images_from_folder(folder):
    """
    Load and preprocess images from a folder.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith('.bmp') or img_path.endswith('.jpg'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
    return images

def prepare_data(face_dir, non_face_dir):
    """
    Prepare the dataset by loading faces and non-faces.
    """
    faces = load_images_from_folder(face_dir)
    non_faces = load_images_from_folder(non_face_dir)
    data = faces + non_faces
    labels = [1] * len(faces) + [0] * len(non_faces)
    return np.array(data).reshape(len(data), -1), np.array(labels)

# Load the trained model
with open('face_detector_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Paths to test data
test_face_dir = os.path.join(DATA_DIR, 'test_cropped_faces')
test_non_face_dir = os.path.join(DATA_DIR, 'test_nonfaces')

# Load and prepare test data
X_test, y_test = prepare_data(test_face_dir, test_non_face_dir)

# Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
