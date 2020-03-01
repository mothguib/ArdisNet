# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler


class ArdisTrainer:

    def __init__(self,
                 model,
                 optimizer="adam",
                 loss="categorical_crossentropy",
                 metrics=None):

        if metrics is None:
            metrics = ["accuracy"]

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # Data generator carries out real-time data augmentation. In
        # particular, that used here is an image data generator that creates
        # more images
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        # Configures the model for training. The Adam optimiser and the
        # cross-entropy loss will be used.
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def run(self, x, y, epochs=30, annealer=None):

        if annealer is None:
            annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

        return self.model. \
            fit_generator(self.datagen.flow(x_train, y_train, batch_size=64),
                          epochs=epochs,
                          steps_per_epoch=x_train.shape[0] // 64,
                          validation_data=(x_val, y_val),
                          callbacks=[annealer],
                          verbose=0)
