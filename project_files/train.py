import numpy as np
import cv2
import os
import joblib

cwd = os.getcwd()
train_face_dir = cwd + "/../training_test_data/training_faces/"
train_nonface_dir = cwd + "/../training_test_data/training_nonfaces/"


class DecisionStump:
    """
    Decision Stump
    """

    def __init__(self):
        self.best_feature = None
        self.best_threshold = None
        self.best_rule = None

    def train(self, X, y, weights):
        num_features = X.shape[1]
        min_error = float("inf")

        for feature in range(num_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                for rule in [1, -1]:
                    predictions = np.where(
                        feature_values * rule < threshold * rule, 1, -1
                    )
                    misclassified = predictions != y
                    weighted_error = np.sum(weights[misclassified])

                    if weighted_error < min_error:
                        min_error = weighted_error
                        self.best_feature = feature
                        self.best_threshold = threshold
                        self.best_rule = rule

    def predict(self, X):
        feature_values = X[:, self.best_feature]
        return np.where(
            feature_values * self.best_rule < self.best_threshold * self.best_rule,
            1,
            -1,
        )


# Function to compute Haar-like features using OpenCV
def compute_haar_features(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    multi_scale = face_cascade.detectMultiScale(
        image, scaleFactor=1.2, minNeighbors=20, minSize=(30, 30)
    )
    faces = []
    for x, y, w, h in multi_scale:
        faces.append([x, y, x + w, y + h])

    return faces


def load_data():
    # Load face images
    train_face_files = os.listdir(train_face_dir)
    x_train_face = []
    for file in train_face_files:
        image = cv2.imread(train_face_dir + file, cv2.IMREAD_GRAYSCALE)
        features = compute_haar_features(image)
        x_train_face.extend(features)

    # Load non-face images

    IMG_SHAPE = (300, 300)
    LENGTH = 100
    STEP = 45

    train_nonface_files = os.listdir(train_nonface_dir)
    x_train_nonface = []
    for file in train_nonface_files:
        image = cv2.imread(train_nonface_dir + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, IMG_SHAPE)
        # inner loop to get sub images

        height, width = image.shape[0], image.shape[1]
        for row in range(0, height - LENGTH + 1, STEP):
            for col in range(0, width - LENGTH + 1, STEP):
                subimage = image[row : row + LENGTH, col : col + LENGTH]
                features = compute_haar_features(subimage)
                
                # returns empty list
                # x_train_nonface.extend(features)
                subwindow_size = 90 
                # replace with [row,col,row+LENGTH,col+LENGTH]?
                x_train_nonface.append([0, 0, subwindow_size, subwindow_size])
               

    # Combine face and non-face data
    x_train = x_train_face + x_train_nonface
    y_train = [1] * len(x_train_face) + [-1] * len(
        x_train_nonface
    )  # 1 for faces, -1 for non-faces

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    num_samples = len(x_train)
    random_indices = np.random.permutation(num_samples)

    x_train = x_train[random_indices]
    y_train = y_train[random_indices]

    print(f'x_train len: {len(x_train)}, y_train len: {len(y_train)}')
    return x_train, y_train


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
    final_prediction = sum(
        alpha * clf.predict(x) for alpha, clf in zip(alpha_values, classifiers)
    )
    return 1 if final_prediction > threshold else -1


if __name__ == "__main__":
    # Main execution
    # train_face_dir = "../training_test_data/training_faces"
    # train_nonface_dir = "../training_test_data/training_nonfaces"
    x_train, y_train = load_data()
    classifiers, alpha_values = adaboost(x_train, y_train, num_classifiers=30)

    # Save the AdaBoost model
    model_filename = "face_detection_model.joblib"
    joblib.dump((classifiers, alpha_values), model_filename)
