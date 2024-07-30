# CatchWithNose

This is the final project for the Image Processing course. "CatchWithNose" is a webcam game where two players compete to catch a target image using a nose landmark. The winner is the player who catches more pictures.

## Overview

The game uses OpenCV to detect and track the nose landmark of each player. The objective is to catch the target image displayed on the screen using the nose landmark as the main point for interaction.
The project includes a simple face predictor using a Haar cascade classifier.

## Implementation

The project utilizes two main techniques:
1. **Haar Cascades**: Used for face detection. Haar cascades are a machine learning-based approach where a cascade function is trained from lots of positive and negative images.
2. **Facial Landmarks**: Used for detecting specific points on the face, such as the nose. The `shape_predictor_68_face_landmarks.dat` file contains a pre-trained model for detecting 68 facial landmarks.

## Prerequisites

Installing OpenCV and other dependencies may take up to 15 minutes. Use the following commands to install the necessary packages:

```bash
pip3 install --upgrade pip
pip3 install opencv-python
pip3 install dlib
pip3 install --upgrade imutils
