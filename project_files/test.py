# Import configuration variables
from config import data_directory, training_directory, code_directory

import os
import cv2
import numpy as np
import joblib
import matplotlib
import math

matplotlib.use("TkAgg")  # Adjust as needed for your environment
data = {
    "clintonAD2505_468x448.jpg": [[146, 226, 96, 176], [56, 138, 237, 312]],
    "DSC01181.JPG": [[141, 181, 157, 196], [144, 184, 231, 269]],
    "DSC01418.JPG": [[122, 147, 263, 285], [129, 151, 305, 328]],
    "DSC02950.JPG": [[126, 239, 398, 501]],
    "DSC03292.JPG": [[92, 177, 169, 259], [122, 200, 321, 402]],
    "DSC03318.JPG": [[188, 246, 178, 238], [157, 237, 333, 414]],
    "DSC03457.JPG": [[143, 174, 127, 157], [91, 120, 177, 206], [94, 129, 223, 257]],
    "DSC04545.JPG": [[56, 86, 119, 151]],
    "DSC04546.JPG": [[105, 137, 193, 226]],
    "DSC06590.JPG": [[167, 212, 118, 158], [191, 228, 371, 407]],
    "DSC06591.JPG": [[180, 222, 290, 330], [260, 313, 345, 395]],
    "IMG_3793.JPG": [
        [172, 244, 135, 202],
        [198, 253, 275, 331],
        [207, 264, 349, 410],
        [160, 233, 452, 517],
    ],
    "IMG_3794.JPG": [
        [169, 211, 109, 148],
        [154, 189, 201, 235],
        [176, 204, 314, 342],
        [170, 206, 445, 483],
        [144, 191, 550, 592],
    ],
    "IMG_3840.JPG": [
        [200, 268, 150, 212],
        [202, 262, 261, 323],
        [222, 286, 371, 430],
        [154, 237, 477, 549],
    ],
    "jackie-yao-ming.jpg": [[45, 77, 93, 124], [61, 91, 173, 200]],
    "katie-holmes-tom-cruise.jpg": [[55, 102, 93, 141], [72, 116, 197, 241]],
    "mccain-palin-hairspray-horror.jpg": [[58, 139, 100, 179], [102, 177, 254, 331]],
    "obama8.jpg": [[85, 157, 109, 180]],
    "phil-jackson-and-michael-jordan.jpg": [[34, 75, 58, 92], [32, 75, 152, 193]],
    "the-lord-of-the-rings_poster.jpg": [
        [222, 267, 0, 35],
        [129, 170, 6, 40],
        [13, 81, 26, 84],
        [22, 92, 120, 188],
        [35, 94, 225, 276],
        [190, 255, 235, 289],
        [301, 345, 257, 298],
    ],
}


# Custom Decision Stump Classifier
class DecisionStump:
    def __init__(self):
        self.best_feature = None
        self.best_threshold = None
        self.best_rule = None

    def predict(self, X):
        feature_values = X[:, self.best_feature]
        return np.where(
            feature_values * self.best_rule < self.best_threshold * self.best_rule,
            1,
            -1,
        )


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
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5)
    )
    features = []
    for x, y, w, h in faces:
        features.append([x, y, x + w, y + h])
    return features


def adaboost_predict(classifiers, alpha_values, X):
    final_predictions = [
        sum(
            alpha * clf.predict(np.array([x]))
            for alpha, clf in zip(alpha_values, classifiers)
        )
        for x in X
    ]
    return [1 if prediction > 0 else -1 for prediction in final_predictions]


def euclidean_distance(point1, point2):
    
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# Assuming 'data' directory is at the same level as 'training_test_data'
model_directory = os.path.join(os.path.dirname(data_directory), "data")
model_filename = os.path.join(model_directory, "face_detection_model.joblib")

output_dir = os.path.join(code_directory, "output")
os.makedirs(output_dir, exist_ok=True)

# Paths to test images are directly inside 'training_test_data'
test_image_dirs = ["test_cropped_faces", "test_face_photos", "test_nonfaces"]

# Output directory (relative path)
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load AdaBoost classifier
ada_boost_classifier = joblib.load(model_filename)

# Processing loop

total_faces = 0

for key in data:
    total_faces += len(data[key])

actual_detected = 0

acc = 0
total_imgs = 0
for dir in test_image_dirs:
    current_test_dir = os.path.join(training_directory, dir)
    test_jpg_files = [
        f
        for f in os.listdir(current_test_dir)
        if f.endswith(".jpg") or f.endswith("JPG")
    ]

    for jpg_file in test_jpg_files:
        total_imgs += 1
        image_path = os.path.join(current_test_dir, jpg_file)
        image = cv2.imread(image_path)

        skin_detected, skin_mask = detect_skin_ycbcr(image)
        gray_skin = cv2.cvtColor(skin_detected, cv2.COLOR_BGR2GRAY)
        faces = compute_haar_features(gray_skin)

        X_test = np.array(faces, dtype=np.float32)
        predictions = adaboost_predict(
            ada_boost_classifier[0], ada_boost_classifier[1], X_test
        )

        centroids_predicted = []
        for face, prediction in zip(faces, predictions):
            if prediction == 1:
                x, y, x2, y2 = face

                # top, bottom, left, right
                # make centroids
                x_center = (x + x2) // 2
                y_center = (y + x2) // 2
                centroids_predicted.append([y_center, x_center])

                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

        # sort centroids based on x axis
        centroids_predicted = sorted(centroids_predicted, key=lambda x: x[1])

        # if we are dealing iwth test_face_photos
        if jpg_file in data:
            centroids = []
            for corners in data[jpg_file]:
                top, bottom, left, right = corners
                x_c = (left + right) // 2
                y_c = (top + bottom) // 2
                centroids.append([y_c, x_c])
            centroids = sorted(centroids, key=lambda x: x[1])

            # compute euclidean distance for each point, if in range of the actual centroid,
            # classifiy as detected
            for index in range(len(centroids_predicted)):
                distance = euclidean_distance(
                    centroids[index], centroids_predicted[index]
                )
                if distance <= 300:
                    actual_detected += 1

        output_path = os.path.join(output_dir, os.path.basename(dir), jpg_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

print(f'Total faces in our test dataset: {total_faces}')
print(f'Actual # of faces detected: {actual_detected}')
print(f'Accuracy of our Model:{actual_detected/total_faces}\n')
print("Processed images saved to the 'output' directory.")
