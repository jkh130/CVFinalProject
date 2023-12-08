import numpy as np
import cv2
import os
import joblib

from config import training_directory, data_directory 

train_face_dir = os.path.join(training_directory, "training_faces/")
train_nonface_dir = os.path.join(training_directory, "training_nonfaces/")


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


def compute_haar_features(image):
    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    frontal_faces = frontal_face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10))

    profile_faces = profile_face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10))

    if len(frontal_faces) == 0:
        faces = profile_faces
    elif len(profile_faces) == 0:
        faces = frontal_faces
    else:
        faces = np.concatenate((frontal_faces, profile_faces), axis=0)

    features = [[x, y, x + w, y + h] for (x, y, w, h) in faces]
    return features


def load_data():
    train_face_files = os.listdir(train_face_dir)
    x_train_face = []
    for file in train_face_files:
        image = cv2.imread(train_face_dir + file, cv2.IMREAD_GRAYSCALE)
        features = compute_haar_features(image)
        x_train_face.extend(features)


    IMG_SHAPE = (300, 300)
    LENGTH = 100
    STEP = 45

    train_nonface_files = os.listdir(train_nonface_dir)
    x_train_nonface = []
    for file in train_nonface_files:
        image = cv2.imread(train_nonface_dir + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, IMG_SHAPE)
        height, width = image.shape[0], image.shape[1]
        for row in range(0, height - LENGTH + 1, STEP):
            for col in range(0, width - LENGTH + 1, STEP):
                subimage = image[row : row + LENGTH, col : col + LENGTH]
                features = compute_haar_features(subimage)

                subwindow_size = 90
            
                x_train_nonface.append([0, 0, subwindow_size, subwindow_size])


    x_train = x_train_face + x_train_nonface
    y_train = [1] * len(x_train_face) + [-1] * len(
        x_train_nonface
    )  

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
    predictions = classifier.predict(X)

    incorrect = predictions != y
    error = np.sum(weights[incorrect]) / np.sum(weights)

    error = min(max(error, 1e-10), 1 - 1e-10)

    alpha = 0.5 * np.log((1 - error) / error)

    return error, alpha


def update_weights(weights, alpha, classifier, X, y):

    predictions = classifier.predict(X)
    correct = predictions == y
    weights *= np.exp(alpha * np.where(correct, -1, 1))
    weights /= np.sum(weights)

    return weights

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

def adaboost_predict(classifiers, alpha_values, x, threshold=0):
    final_prediction = sum(
        alpha * clf.predict(x) for alpha, clf in zip(alpha_values, classifiers)
    )
    return 1 if final_prediction > threshold else -1


if __name__ == "__main__":
    x_train, y_train = load_data()
    classifiers, alpha_values = adaboost(x_train, y_train, num_classifiers=1)

    model_filename = os.path.join(data_directory, "face_detection_model.joblib")
    joblib.dump((classifiers, alpha_values), model_filename)
