# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def accuracy(result, y):
    m = tf.keras.metrics.Accuracy()
    _ = m.update_state(np.argmax(result, axis=1), np.argmax(y, axis=1))

    return m.result().numpy()