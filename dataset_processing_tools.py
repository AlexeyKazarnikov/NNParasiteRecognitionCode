import cv2 as cv
import numpy as np

from os import listdir
from os.path import isfile, join

import image_processing_tools as imtools


# this function loads all files from the selected folder (it is assumed that all files are images)
def load_dataset_images(folder_path):
    image_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    images = []
    for image_file in image_files:
        image = cv.imdecode(np.fromfile(join(folder_path, image_file), dtype=np.uint8), cv.IMREAD_COLOR)
        images.append(image)
    return images


# this function implements a data augmentation scheme for the set of images
def run_data_augmentation(images, sample_number, resolution, eig_vals, eig_vecs):
    output_images = np.zeros((len(images) * sample_number, resolution[0], resolution[1], 3), dtype=np.single)
    for i in range(len(images)):
        for j in range(sample_number):
            output_images[i * sample_number + j, :, :, :] = \
                imtools.transform_image(images[i], eig_vals, eig_vecs, resolution)
    return output_images


# this function implements the generation of phantoms (artificial examples) for the set of images
def run_phantom_generation(parasite_images, background_images, sample_number, resolution, eig_vals, eig_vecs):
    output_images = np.zeros((sample_number, resolution[0], resolution[1], 3), dtype=np.single)
    for i in range(sample_number):
        i0 = np.random.randint(len(background_images))
        i1 = np.random.randint(len(parasite_images))
        input_phantom = imtools.generate_phantom(background_images[i0], parasite_images[i1])
        output_phantom = imtools.transform_image(input_phantom, eig_vals, eig_vecs, resolution)
        output_images[i, :, :, :] = output_phantom
    return output_images


# this function is used to construct a training dataset of augmented data (no phantoms)
def construct_training_data(images_1, images_2, resolution, requested_dataset_size, eig_vals, eig_vecs):
    img_number_1 = len(images_1)
    img_number_2 = len(images_2)

    sample_number_1 = round(requested_dataset_size / (2 * img_number_1))
    sample_number_2 = round(requested_dataset_size / (2 * img_number_2))

    actual_dataset_size = img_number_1 * sample_number_1 + img_number_2 * sample_number_2
    image_data = np.zeros((actual_dataset_size, resolution[0], resolution[1], 3), dtype=np.single)
    label_data = np.zeros((actual_dataset_size, 2), dtype=np.single)

    image_data[0:img_number_1 * sample_number_1, :, :, :] = \
        run_data_augmentation(images_1, sample_number_1, resolution, eig_vals, eig_vecs)
    label_data[0:img_number_1 * sample_number_1, 0] = 1

    image_data[img_number_1 * sample_number_1:, :, :, :] = \
        run_data_augmentation(images_2, sample_number_2, resolution, eig_vals, eig_vecs)
    label_data[img_number_1 * sample_number_1:, 1] = 1

    index_permutation = np.random.permutation(actual_dataset_size)
    image_data = image_data[index_permutation, :, :, :]
    label_data = label_data[index_permutation, :]

    return image_data, label_data


# this function is used to construct a training dataset of augmented data (with phantoms)
def construct_training_data_phantoms(images_1, images_2, resolution, requested_dataset_size, eig_vals, eig_vecs, background_images):
    img_number_1 = len(images_1)
    img_number_2 = len(images_2)

    sample_number_1 = round(requested_dataset_size / (4 * img_number_1))
    sample_number_2 = round(requested_dataset_size / (2 * img_number_2))

    actual_dataset_size = 2 * img_number_1 * sample_number_1 + img_number_2 * sample_number_2
    image_data = np.zeros((actual_dataset_size, resolution[0], resolution[1], 3), dtype=np.single)
    label_data = np.zeros((actual_dataset_size, 2), dtype=np.single)

    image_data[0:img_number_1 * sample_number_1, :, :, :] = \
        run_data_augmentation(images_1, sample_number_1, resolution, eig_vals, eig_vecs)
    label_data[0:img_number_1 * sample_number_1, 0] = 1

    image_data[img_number_1 * sample_number_1:2 * (img_number_1 * sample_number_1), :, :, :] = \
        run_phantom_generation(images_2, background_images, img_number_1 * sample_number_1, resolution, eig_vals, eig_vecs)
    label_data[img_number_1 * sample_number_1:2 * (img_number_1 * sample_number_1), 0] = 1

    image_data[2 * (img_number_1 * sample_number_1):, :, :, :] = \
        run_data_augmentation(images_2, sample_number_2, resolution, eig_vals, eig_vecs)
    label_data[2 * (img_number_1 * sample_number_1):, 1] = 1

    index_permutation = np.random.permutation(actual_dataset_size)
    image_data = image_data[index_permutation, :, :, :]
    label_data = label_data[index_permutation, :]

    return image_data, label_data
