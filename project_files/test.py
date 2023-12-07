import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from  src.detect_skin import detect_skin
import joblib
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc., depending on your environment
import matplotlib.pyplot as plt

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

def detect_skin_hist(image):
    
    #lower score from detection means higher chance it is a face
    #Does alright with these loaded values
    # but these values are from images that were different
    negative_histogram = np.load("data/negative_histogram.npy")
    positive_histogram = np.load("data/positive_histogram.npy")
    detection = detect_skin(image, positive_histogram, negative_histogram)
    plt.imshow(detection)
    plt.show()

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

model_filename = 'face_detection_model.joblib'
ada_boost_classifier = joblib.load(model_filename)

test_image_dir = '../training_test_data/test_face_photos'
output_dir = '../output/'
os.makedirs(output_dir, exist_ok=True)

test_jpg_files = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg') or f.endswith('JPG')]

for jpg_file in test_jpg_files:
    image_path = os.path.join(test_image_dir, jpg_file)
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

    output_path = os.path.join(output_dir, jpg_file)
    cv2.imwrite(output_path, image)

print("Processed images saved to the 'output' directory.")
