# NNParasiteRecognitionCode
 This repository contains source code for the paper "Development of a neural network algorithm for the recognition of monogeneans of the order Dactylogyridea", to appear in the Journal of Siberian Federal University. Biology. The code is implemented in Python programming language. The VGG-16 convolutional neural network is trained using the TensorFlow library, while the OpenCV library is used for image processing.
 
 The biological data collected for this work are publicly available at the Yandex Drive cloud storage platform: https://disk.yandex.ru/d/rp4w1iWQnT88bg
 
 ## Contents
 There are two main scripts in the repository:
 1. "train_network.py" implements the training of the VGG-16 convolutional neural network for the recognition of the members of the order D. on the images taken from the ocular of the microscope by smartphone camera. Before using this script, please download the provided parasite image datasets using the link above and locate them in some folder on your local machine.
 2. "test_network.py" implements the testing of the trained VGG-16 CNN on the elements of the validation and testing sets.
 
 Additionally, please use the script "estimate_colour_covariance.py" to estimate the colour covariance matrix of the image datasets (if needed).
