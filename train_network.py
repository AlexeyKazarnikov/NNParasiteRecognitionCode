# This script implements the training of the VGG-16 convolutional neural network
# for the recognition of the members of the order D. on the images taken from the
# ocular of the microscope by smartphone camera

from keras import optimizers
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import load_model
from threading import Thread

import gc
import tensorflow as tf

from dataset_processing_tools import *
from vgg16_norm import vgg16normnet2_keras

# fixed resolution of the input image for the neural network. The value must be exactly the same
# as the value used during the testing process!
image_target_resolution = (224, 224)
# batch sizes for training and validation sets
training_batch_size = 16
validation_batch_size = 16
# dropout rate (used in the fully-connected layers)
dropout_rate = 0.6
# L2-regularisation rate
regularisation_rate = 1e-5
# learning rate of the SGD method
learning_rate = 1e-4
# number of (augmented!) images in one epoch for training and validation sets
epoch_dataset_size = 1500
validation_dataset_size = 500
# number of epochs to run the training algorithm
epoch_number = 3000

# name of the file, where network weights are stored. Should be empty for new training process, or
# point out to the intermediate data if the training process is continued
input_file_name = ''
# name of the file where output network weights will be stored. This file is being overwritten after
# each epoch
output_file_name = 'nn_weights.h5'
# best candidate determines the state when the highest accuracy on the validation set is achieved. Best
# candidate is updated (when needed) after each epoch. Might be helpful for implementing early stopping
# of the network (to avoid overfitting)
best_candidate_file_name = 'best_candidate.h5'

# during the training process the information about the values of loss and accuracy functions on the
# training and validation sets is saved in this text file. The contents of the file are overwritten after
# each epoch
history_file_name = 'history.txt'


# we start from loading training and validation data
print('Loading image data...')

# in folder names prefix [0] means the members of the order D. ([1] - non-members), while in the code
# [1] stands for non-members and [2] for members. The old notation was kept to avoid potential errors
# during the refactoring process
print('Training set...')
training_images_1 = load_dataset_images('TrainingSet/Class1')
training_images_2 = load_dataset_images('TrainingSet/Class0')

print('Validation set...')
validation_images_1 = load_dataset_images('ValidationSet/Class1')
validation_images_2 = load_dataset_images('ValidationSet/Class0')

# background images are used to generate phantoms during the data augmentation process
print('Background images...')
background_images = load_dataset_images('BackgroundImages')

print('Data loading completed!')

# loading the covariance matrix for colour channels computed over the elements of the training set. This
# value is needed for data augmentation
pixels_cov = np.load('color_cov.npy')
eig_vals, eig_vecs = np.linalg.eigh(pixels_cov)

# before the training we create the augmented samples for the training and validation sets
print('Beginning data augmentation (initial)...')
image_data, label_data = \
        construct_training_data_phantoms(
            training_images_1, training_images_2, image_target_resolution, epoch_dataset_size, eig_vals, eig_vecs, background_images)

# data augmentation for the validation set is done only once because of performance reasons
print('Data augmentation completed!')
val_image_data, val_label_data = \
        construct_training_data(
            validation_images_1, validation_images_2, image_target_resolution, validation_dataset_size, eig_vals, eig_vecs)

# compiling the model or restoring it from saved weights
if not input_file_name:
    model = vgg16normnet2_keras(image_target_resolution[0], dropout_rate)
    sgd = optimizers.SGD(learning_rate=learning_rate, decay=5e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print("Model compiled.")
else:
    model = load_model(input_file_name)
    keras_backend.set_value(model.optimizer.learning_rate, learning_rate)
    print("Model restored.")

# initialising helper structures for the training process
# the information about loss and accuracy functions
history_data = []
# the maximum value of accuracy achieved at the validation set. Needed for updating the best candidate
best_val_accuracy = 0
# image and label data, where the data augmentation thread stores the results
temp_image_data = np.zeros(image_data.shape)
temp_label_data = np.zeros(label_data.shape)

# function, implementing data augmentation for the training set (in asynchronous regime)
def data_augmentation_task(
        training_images_1,
        training_images_2,
        image_target_resolution,
        epoch_dataset_size,
        eig_vals,
        eig_vecs,
        background_images,
        output_image_data,
        output_label_data):

    image_data, label_data = \
        construct_training_data_phantoms(
            training_images_1, training_images_2, image_target_resolution, epoch_dataset_size, eig_vals, eig_vecs,background_images)
    output_image_data[:, :, :, :] = image_data
    output_label_data[:, :] = label_data

# defining a thread for asynchronous data augmentation
data_augmentation_thread = Thread(
    target=data_augmentation_task,
    args=(training_images_1, training_images_2, image_target_resolution,
          epoch_dataset_size, eig_vals, eig_vecs, background_images, temp_image_data, temp_label_data)
    )

# starting a thread for data augmentation and immediately proceeding further to training
print('Thread: Beginning data augmentation (concurrent)...')
data_augmentation_thread.start()

# entering the main training loop
for epoch in range(epoch_number):
    print(f'Epoch {epoch}')

    # if the thread has already finished working, we replace our current training set with new
    # augmented data (and start the new thread); otherwise we continue with the current dataset
    if not data_augmentation_thread.is_alive():
        print('Thread: Data augmentation completed!')
        image_data = temp_image_data.copy()
        label_data = temp_label_data.copy()
        print('Data exchange completed!')

        print('Thread: Beginning data augmentation (concurrent)...')
        data_augmentation_thread = Thread(
            target=data_augmentation_task,
            args=(training_images_1, training_images_2, image_target_resolution,
                  epoch_dataset_size, eig_vals, eig_vecs, background_images, temp_image_data, temp_label_data)
        )
        data_augmentation_thread.start()

    # network training
    history = model.fit(
        x=image_data,
        y=label_data,
        batch_size=training_batch_size,
        validation_data=(val_image_data, val_label_data),
        verbose=1)

    # updating the statistics
    history_data.append(
        [
            history.history['accuracy'][0],
            history.history['loss'][0],
            history.history['val_accuracy'][0],
            history.history['val_loss'][0]
        ])

    if history_file_name:
        np.savetxt(history_file_name, np.array(history_data), delimiter=',')

    # saving the model weights
    model.save(output_file_name)
    print("Intermediate data saved.")

    # updating the best candidate information
    epoch_val_accuracy = history.history['val_accuracy'][0]
    if epoch_val_accuracy > best_val_accuracy:
        model.save(best_candidate_file_name)
        best_val_accuracy = epoch_val_accuracy
        print("Best candidate updated.")

    tf.keras.backend.clear_session()
    gc.collect()

