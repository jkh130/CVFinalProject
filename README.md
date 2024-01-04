**Computer Vision Face Detection System**

**Introduction**

Face detection system developed as part of the Computer vision final project. It utilizes AdaBoost, skin detection algorithms, and classifier cascades to identify human faces in color images. The project's objective is to integrate AdaBoost for feature selection and classification effectively, along with skin detection algorithms to improve efficiency and accuracy.

**Implementation**

**AdaBoost**
Custom *'DecisionStump'* Class: Serves as the weak classifier to identify the best threshold, feature, and rule for classification.
Training Process: Iterative selection of weak classifiers, assigning weights based on accuracy, and storing classifiers with their corresponding alpha values.

**Skin Detection**

detect_skin_ycbcr Function: Implements skin detection using the YCbCr color space.
YCbCr Color Space: Chosen for its effectiveness in diverse skin tones and lighting conditions.
Haar Feature Computation
compute_haar_features Function: Utilizes OpenCV's pre-trained Haar cascade classifiers for face detection.
Parameters Tuning: scaleFactor, minNeighbors, and minSize are tuned to optimize detection accuracy.
Classifier Cascades
Cascade Structure: Organizes classifiers in a way that simple classifiers reject negative instances early, enhancing detection speed and accuracy.
Parameter Justification
Haar Feature Parameters: scaleFactor, minNeighbors, and minSize parameters are optimized for detection accuracy and computational efficiency.
AdaBoost Parameters: Number of classifiers and weak classifier parameters are configured for performance enhancement.
Skin Detection Thresholds: Specific thresholds in the YCbCr color space for accurate skin pixel differentiation.
Challenges Overview

**Skin Detection**

Accuracy: Initial challenges in accurately detecting skin tones, adjusted through empirical observation and research.
Dynamic Adjustment: Trade-offs between accuracy and generalization in different images.

**Haar Feature Selection**

Computational Complexity: Increased complexity and accuracy issues when combining multiple Haar cascade classifiers.
