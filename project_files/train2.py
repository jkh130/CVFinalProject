import os
import cv2
import numpy as np
from boosting import (
    eval_weak_classifier,
    generate_classifier,
    integral_image,
    adaboost,
    boosted_predict,
)
import time
import random

# Cascading Classifier - multiple layers compromised of classifiers 
# 1. make random classifiers for each layer: amouut 500
# 2. use adaboosint to select the best # , top 10,15,20 etc
# 3 apply the classifier to each subwindow
# 4. use skin detection, if its not skin, just skip


cwd = os.getcwd()
train_face_dir = cwd + "/../training_test_data/training_faces/"
train_nonface_dir = cwd + "/../training_test_data/training_nonfaces/"



def parse_data():
    """
    returns:
        train_faces: (3047, 100, 100, 3)
        train_non_faces: (3250, 100, 100, 3)

    """
    LENGTH = 100
    IMG_SHAPE = (300, 300)
    STEP = 45
    
    # train_faces
    files = os.listdir(train_face_dir)
    train_faces = [cv2.imread(train_face_dir + file) for file in files]
    train_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in train_faces]
    train_faces = np.array(train_faces)
    
    # train_non_faces
    files = os.listdir(train_nonface_dir)
    images = [cv2.imread(train_nonface_dir + file) for file in files]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    images = [cv2.resize(img, IMG_SHAPE) for img in images]
    
    train_non_faces = []
    for img in images:
        height, width = img.shape[0], img.shape[1]
        for row in range(0, height - LENGTH + 1, STEP):
            for col in range(0, width - LENGTH + 1, STEP):
                train_non_faces.append(img[row : row + LENGTH, col : col + LENGTH])
        # break
    train_non_faces = np.array(train_non_faces)
    
    labels = np.array([1] * train_faces.shape[0] + [-1] * train_non_faces.shape[0])

    training_dataset = np.concatenate((train_faces, train_non_faces))
    print(f'train shape:{train_faces.shape}')
    print(f'train non face shape: {train_non_faces.shape}')
    print(f'trian dataset: {training_dataset.shape}\n')

    return train_faces, train_non_faces, training_dataset, labels


def calculate_image_integrals(train_faces, train_non_faces):
    face_integrals = []
    nonface_integrals = []

    # Compute the integral images for all faces
    for face in train_faces:
        face_integrals.append(integral_image(face))

    # Compute the integral images for all non-faces
    for nonface in train_non_faces:
        nonface_integrals.append(integral_image(nonface))

    face_integrals = np.array(face_integrals)
    nonface_integrals = np.array(nonface_integrals)

    image_integrals = np.concatenate((face_integrals, nonface_integrals), axis=0)
    # same as original?
    labels = np.array([1] * train_faces.shape[0] + [-1] * train_non_faces.shape[0])

    print(f"Shape of the image_integrals matrix: {image_integrals.shape}")

    return image_integrals, labels
    

if __name__ == "__main__":
    train_faces, train_non_faces, training_dataset, labels = parse_data()

    # filter size - Has to be same size as input img for generate classfiers
    face_vertical = 100
    face_horizontal = 100
    """
    wc = generate_classifier1(face_vertical, face_horizontal)
    for key in wc:
        print(key,wc[key])
    
    index = 0
    face = training_dataset[index]
    integral_face = integral_image(face)
    response = eval_weak_classifier(wc, integral_face)
    print(f"Response1: {response}")"""

    NUM_CLASSFIERS = 100

    # 2
    weak_classifiers = [
        generate_classifier(face_vertical, face_horizontal)
        for _ in range(NUM_CLASSFIERS)
    ]
    print(f"Number of weak classifiers generated: {len(weak_classifiers)}")

    # 3 PRE-PROCESSING - computing all image integrals
    # responses != image_integrals
    image_integrals, labels = calculate_image_integrals(train_faces, train_non_faces)

    # 4 apply weak classifiers to integral images

    example_number = image_integrals.shape[0]
    classifier_number = len(weak_classifiers)

    print(f'example_number: {example_number} classifier_number:{classifier_number}\n')

    # Initialize an array to hold the responses
    # classifiers
    responses = np.zeros((example_number, classifier_number))

    # Loop through each example and classifier
    for example in range(example_number):
        integral = image_integrals[example, :, :]
        for feature in range(classifier_number):
            classifier = weak_classifiers[feature]
            responses[example, feature] = eval_weak_classifier(classifier, integral)

    print(f'response shape:{responses.shape}')
    
    
    # 5 select random image and rand class and see response values

    classifier_index = random.randint(0, classifier_number - 1)
    wc = weak_classifiers[classifier_index]

    # Choose a training image
    example_index = random.randint(0, example_number - 1)
    example_integral = image_integrals[example_index, :, :]

    # See the precomputed response
    print(f"Selected classifier index: {classifier_index}, selected image index: {example_index}")
    print(f"Precomputed response at (example_index, classifier_index): {responses[example_index, classifier_index]}")
    print(f"Evaluated weak classifier response: {eval_weak_classifier(wc, example_integral)}")

    # 6 WEIGHTS
    # weights for each clasifier
    
    weights = np.ones(example_number) / example_number

    # # AFTER ADABOOSTING

    print("\n\n---------ADABOOSTIN---------\n\n")

    print(f'responses.shape: {responses.shape} labels.shape: {labels.shape} weights.shape: {weights.shape}')
    # # Define the number of weak classifiers to to select
    BEST_CLASSIFIERS = 15

    # Run the AdaBoost algorithm

    print(labels.shape)

    boosted_classifier = adaboost(responses, labels, BEST_CLASSIFIERS)

    ### PREDICTIOM
    # Predict the label of the 200th face example
    prediction = boosted_predict(train_faces[200, :, :], boosted_classifier, weak_classifiers, NUM_CLASSFIERS)
    print(f"Prediction: {prediction}")

    # Predict the label of the 500th non-face example
    prediction = boosted_predict(train_non_faces[500, :, :], boosted_classifier, weak_classifiers, NUM_CLASSFIERS)
    print(f"Prediction: {prediction}")
   
   
   

    # # EVALUATIOM
    print("\n\n---------Evaluation---------\n\n")
    # A value further away from zero is better for classification
   # Evaluate the boosted classifier on the faces and non-faces
    face_predictions = boosted_predict(train_faces, boosted_classifier, weak_classifiers, NUM_CLASSFIERS)
    nonface_predictions = boosted_predict(train_non_faces, boosted_classifier, weak_classifiers, NUM_CLASSFIERS)

    FACE_THRESH = 0 # A face is detected if the response is greater than or equal to this threshold

    # Calculate the accuracy of the predictions
    face_accuracy = np.sum(face_predictions >= FACE_THRESH) / len(face_predictions)
    nonface_accuracy = np.sum(nonface_predictions < FACE_THRESH) / len(nonface_predictions)

    # Print the results
    print(f"Face accuracy: {face_accuracy}")
    print(f"Non-face accuracy: {nonface_accuracy}")
