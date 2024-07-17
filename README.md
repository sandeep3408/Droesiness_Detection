Project Overview
This project aims to detect drowsiness in real-time using computer vision techniques. It utilizes an eye dataset along with Haar Cascade classifiers for frontal face and eye detection. The primary goal is to alert drivers when signs of drowsiness are detected to prevent accidents caused by falling asleep at the wheel.

The dataset used in this project consists of images and videos containing faces and eyes, which are used to train and test the drowsiness detection system. The dataset includes:

Eye Dataset: A collection of images with open and closed eyes.
Haar Cascade Classifiers: Pre-trained models for detecting frontal faces and eyes.
Installation

The drowsiness detection system follows these steps:

Face Detection:

Use the Haar Cascade frontal face classifier to detect faces in the video stream.
Eye Detection:

Use the Haar Cascade eye classifier to detect eyes within the detected face region.
Eye Aspect Ratio (EAR):

Calculate the Eye Aspect Ratio (EAR) to determine if the eyes are open or closed. If the EAR is below a certain threshold, it indicates that the eyes are closed.
Drowsiness Alert:

Monitor the duration for which the eyes remain closed. If the eyes remain closed for a prolonged period, trigger a drowsiness alert.
Results
The results section will include:

Accuracy and performance metrics of the drowsiness detection system.
Examples of real-time detection and alerts.
Analysis of false positives and false negatives.
Contributing
Contributions are welcome! If you have any suggestions, bug reports, or improvements, feel free to create an issue or submit a pull request.

