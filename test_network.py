# This script implements the testing of the trained VGG-16 CNN on the elements
# of the validation and testing sets

from collections import namedtuple
from keras.models import load_model

from dataset_processing_tools import *
import image_processing_tools as imtools

# fixed resolution of the input image for the neural network. The value must be exactly the same
# as the value used during the training process!
image_target_resolution = (224, 224)

# name of the file, where network weights are stored
input_file_name = 'vgg16_pretrained_weights.h5'

# loading the model using the built-in Keras function
model = load_model(input_file_name)
print("Model restored.")

# loading a covariance matrix of colour channels, computed using
# the elements of the training set (to be used for the creation of augmented examples)
pixels_cov = np.load('color_cov.npy')
eig_vals, eig_vecs = np.linalg.eigh(pixels_cov)

# creating a confusion matrix with elements:
#   [[Total population (TP), Predicted positive (PP), Predicted negative (PN)], ...
#    [Actual positive (AP), True positive (TP), False negative (FN)], ...
#    [Actual negative (AN), False positive (FP), True negative (TN)]]
confusion_matrix = np.zeros((3, 3))

Settings = namedtuple('Settings', 'image_target_resolution, eig_vals, eig_vecs')
model_settings = Settings(image_target_resolution, eig_vals, eig_vecs)


def process_image(image_path, data_class, model, settings, confusion_matrix):
    # loading image from file
    image = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), cv.IMREAD_COLOR)
    # creating a testing set of augmented images
    augmented_images = run_data_augmentation(
        [image],
        100,
        settings.image_target_resolution,
        settings.eig_vals,
        settings.eig_vecs
    )
    # applying the recognition algorithm to augmented images
    labels = model.predict(augmented_images, verbose=0)
    averaged_label = np.sum(labels, axis=0)

    # making decision
    if data_class == 0:
        confidence_level = np.sum(labels[:, 0] <= labels[:, 1])
        is_correct = averaged_label[0] <= averaged_label[1]
    else:
        confidence_level = np.sum(labels[:, 0] > labels[:, 1])
        is_correct = averaged_label[0] > averaged_label[1]

    # updating values of the confusion matrix
    confusion_matrix[0, 0] += 1  # total population
    if data_class == 0:
        confusion_matrix[1, 0] += 1  # actual positive
        if is_correct:
            confusion_matrix[0, 1] += 1  # predicted positive
            confusion_matrix[1, 1] += 1  # true positive
        else:
            confusion_matrix[0, 2] += 1  # predicted negative
            confusion_matrix[1, 2] += 1  # false negative
    else:
        confusion_matrix[2, 0] += 1  # actual negative
        if is_correct:
            confusion_matrix[0, 2] += 1  # predicted negative
            confusion_matrix[2, 2] += 1  # true negative
        else:
            confusion_matrix[0, 1] += 1  # predicted positive
            confusion_matrix[2, 1] += 1  # false positive

    # printing statistics
    print(
        f'Image: {image_path}, correct decision: {is_correct}, confidence: {confidence_level}%')


def process_sub_folder(
        sub_folder_name,
        model,
        settings,
        confusion_matrix
):
    print('Processing data from the Class 0 (a member of the order D.)...')
    folder_path = sub_folder_name + '\\Class0'
    image_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for image_file in image_files:  # iterating over all file names in the selected folder
        image_path = join(folder_path, image_file)
        process_image(image_path, 0, model, settings, confusion_matrix)

    print('Processing data from the Class 1 (not a member of the order D.)...')
    folder_path = sub_folder_name + '\\Class1'
    image_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for image_file in image_files:  # iterating over all file names in the selected folder
        image_path = join(folder_path, image_file)
        process_image(image_path, 1, model, settings, confusion_matrix)


# processing images of the validation set
print('Processing elements of the validation set...')
process_sub_folder('ValidationSet', model, model_settings, confusion_matrix)
print('Done!')
# processing images of the testing set
print('Processing elements of the testing set...')
process_sub_folder('TestingSet', model, model_settings, confusion_matrix)
print('Done!')

print('Confusion matrix:')
print(f'TP: {confusion_matrix[0, 0]}   PP: {confusion_matrix[0, 1]}   PN: {confusion_matrix[0, 2]}')
print(f'AP: {confusion_matrix[1, 0]}   TP: {confusion_matrix[1, 1]}   FN: {confusion_matrix[1, 2]}')
print(f'AN: {confusion_matrix[2, 0]}   FP: {confusion_matrix[2, 1]}   TN: {confusion_matrix[2, 2]}')
print(f'Accuracy:{(confusion_matrix[1, 1] + confusion_matrix[2, 2]) / confusion_matrix[0, 0]}')
