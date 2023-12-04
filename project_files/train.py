import numpy as np
import cv2
import os
import joblib
class DecisionStump:
    def __init__(self):
        self.best_feature = None
        self.best_threshold = None
        self.best_rule = None

    def train(self, X, y, weights):
        num_features = X.shape[1]
        min_error = float('inf')

        for feature in range(num_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                for rule in [1, -1]:
                    predictions = np.where(feature_values * rule < threshold * rule, 1, -1)
                    misclassified = predictions != y
                    weighted_error = np.sum(weights[misclassified])

                    if weighted_error < min_error:
                        min_error = weighted_error
                        self.best_feature = feature
                        self.best_threshold = threshold
                        self.best_rule = rule

    def predict(self, X):
        feature_values = X[:, self.best_feature]
        return np.where(feature_values * self.best_rule < self.best_threshold * self.best_rule, 1, -1)
    

# Function to compute Haar-like features using OpenCV
def compute_haar_features(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=20, minSize=(30, 30))
    features = []
    for (x, y, w, h) in faces:
        features.append([x, y, x + w, y + h])
    return features

# Load and preprocess the data
def load_data(train_face_dir, train_nonface_dir):
    # Load face images
    train_face_files = [f for f in os.listdir(train_face_dir) if f.endswith('.bmp')]
    X_train_face = []
    for bmp_file in train_face_files:
        image_path = os.path.join(train_face_dir, bmp_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = compute_haar_features(image)
        X_train_face.extend(features)

    # Load non-face images
    train_nonface_files = [f for f in os.listdir(train_nonface_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
    X_train_nonface = []
    for jpg_file in train_nonface_files:
        image_path = os.path.join(train_nonface_dir, jpg_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        X_train_nonface.append([0, 0, image.shape[1], image.shape[0]])  # Use the whole image as a non-face feature

    # Combine face and non-face data
    X_train = X_train_face + X_train_nonface
    y_train = [1] * len(X_train_face) + [-1] * len(X_train_nonface)  # 1 for faces, -1 for non-faces

    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32)


def initialize_weights(n):
    return np.ones(n) / n

def train_weak_classifier(X, y, weights):
    stump = DecisionStump()
    stump.train(X, y, weights)
    return stump



def calculate_error_and_alpha(classifier, X, y, weights):
    # Make predictions using the classifier
    predictions = classifier.predict(X)

    # Determine which predictions are incorrect
    incorrect = predictions != y

    # Calculate weighted error
    error = np.sum(weights[incorrect]) / np.sum(weights)

    # Avoid division by zero; handle edge cases
    error = min(max(error, 1e-10), 1 - 1e-10)

    # Calculate alpha
    alpha = 0.5 * np.log((1 - error) / error)

    return error, alpha


def update_weights(weights, alpha, classifier, X, y):
    # Make predictions using the classifier
    predictions = classifier.predict(X)

    # Determine whether each prediction is correct
    correct = predictions == y

    # Update weights: increase for misclassified, decrease for correctly classified
    weights *= np.exp(alpha * np.where(correct, -1, 1))

    # Normalize the weights so they sum up to 1
    weights /= np.sum(weights)

    return weights

# AdaBoost training process
def adaboost(X, y, num_classifiers):
    n = len(y)
    weights = initialize_weights(n)
    classifiers = []
    alpha_values = []

    for _ in range(num_classifiers):
        classifier = train_weak_classifier(X, y, weights)
        error, alpha = calculate_error_and_alpha(classifier, X, y, weights)
        weights = update_weights(weights, alpha, classifier, X, y)
        classifiers.append(classifier)
        alpha_values.append(alpha)

    return classifiers, alpha_values

# Prediction using AdaBoost
def adaboost_predict(classifiers, alpha_values, x, threshold=0):
    final_prediction = sum(alpha * clf.predict(x) for alpha, clf in zip(alpha_values, classifiers))
    return 1 if final_prediction > threshold else -1

# Main execution
train_face_dir = 'training_test_data/training_faces'
train_nonface_dir = 'training_test_data/training_nonfaces'
X_train, y_train = load_data(train_face_dir, train_nonface_dir)
classifiers, alpha_values = adaboost(X_train, y_train, num_classifiers=30)

# Save the AdaBoost model
model_filename = 'face_detection_model.joblib'
joblib.dump((classifiers, alpha_values), model_filename)

