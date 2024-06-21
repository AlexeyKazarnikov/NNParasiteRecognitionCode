# This module defines the architecture of the VGG-16 CNN, which is used in the current work

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


def vgg16normnet2_keras(image_resolution, dropout_rate):
    model = Sequential()

    # Unit 1
    model.add(
        Conv2D(input_shape=(image_resolution, image_resolution, 3),
               filters=64, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Unit 2
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Unit 3
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Unit 4
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Unit 5
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=2, activation="softmax"))

    return model
