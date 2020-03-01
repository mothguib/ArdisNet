# -*- coding: utf-8 -*-

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization


class ArdisNet(Sequential):

    def __init__(self):
        super(ArdisNet, self).__init__()

        self.add(Conv2D(32, kernel_size=3, activation='relu',
                        input_shape=(28, 28, 1)))
        self.add(BatchNormalization())
        self.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(32, kernel_size=5, strides=2, padding='same',
                        activation='relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(64, kernel_size=3, activation='relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, kernel_size=3, activation='relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                        activation='relu'))
        self.add(BatchNormalization())
        self.add(Dropout(0.4))

        self.add(Conv2D(128, kernel_size=4, activation='relu'))
        self.add(BatchNormalization())
        self.add(Flatten())
        self.add(Dropout(0.4))

        self.add(Dense(10, activation='softmax'))
