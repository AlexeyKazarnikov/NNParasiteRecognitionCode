# This script estimates the covariance of colour channels from the images of the training and validation sets.

import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join


# this function iterates over all images in folder and extract all non-transparent pixels
def process_dataset_folder(folder_path):
    image_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    pixel_data = np.zeros((0, 3))
    for image_file in image_files:
        image = cv.imread(join(folder_path, image_file))
        image_float = image.astype(np.single) / 255

        # reshaping 3D image into an array of pixels
        pixels = image_float.reshape(-1, 3)
        pixels_abs_values = np.sum(pixels, axis=1)
        positive_mask = (pixels_abs_values > 0)
        pixels = pixels[positive_mask, :]
        pixel_data = np.append(pixel_data, pixels, axis=0)
    return pixel_data


# this function processes the folder containing the image dataset
def process_image_dataset(dataset_path):
    pixel_data_1 = process_dataset_folder(join(dataset_path, 'Class0'))
    pixel_data_2 = process_dataset_folder(join(dataset_path, 'Class1'))
    return np.append(pixel_data_1, pixel_data_2, axis=0)


print('Beginning the estimation of the colour covariance matrix...')

training_pixel_data = process_image_dataset('TrainingSet')
validation_pixel_data = process_image_dataset('ValidationSet')

complete_pixel_data = np.append(training_pixel_data, validation_pixel_data, axis=0)

# computing covariance matrix of the pixel values
complete_pixel_data_cov = np.cov(complete_pixel_data, rowvar=False)
np.save('color_cov.npy', complete_pixel_data_cov)

print('Estimation completed!')
