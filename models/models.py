import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Lambda,
    ZeroPadding2D,
    MaxPool2D,
    BatchNormalization
)
from keras import  losses
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class SimpleBirdNet:
    def __init__(self, width=224, height=224, channels=3):
        self.model = self.init_model(width, height, channels)

    def init_model(self, width, height, channels):
        model = Sequential()

        model.add(Conv2D(input_shape=(width, height, channels), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=3))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), activation="relu"))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters=128, kernel_size=(1, 1), activation="relu"))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=512, kernel_size=(5, 5), activation="relu"))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dense(units=256, activation="relu"))

        model.add(Dense(units=40, activation="softmax"))

        return model


class vgg16:
    def __init__(self, width=224, height=224, channels=3):
        self.model = self.init_model(width, height, channels)

    def init_model(self, width, height, channels):
        vgg = Sequential()
        conv_base = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
        vgg.add(conv_base)
        vgg.add(Flatten())
        vgg.add(Dense(256, activation="relu"))
        vgg.add(Dense(128, activation="relu"))
        vgg.add(Dense(units=40, activation="softmax"))

        return vgg

class alexnet:
    def __init__(self, width=224, height=224, channels=3):
        self.model = self.init_model(width, height, channels)

    def init_model(self, width, height, channels):
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=(11, 11), strides=(4, 4), padding='same', activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(4096, input_shape=(32, 32, 3,), activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(4096, activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(1000, activation="relu"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(40))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))