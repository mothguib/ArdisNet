# -*- coding: utf-8 -*-

import os
import numpy as np

from ardis import PKG_ROOT

ARDIS_PATH = "datasets/dataset-4/"


def load_data():
    return (np.loadtxt(os.path.join(PKG_ROOT, ARDIS_PATH,
                                    "ARDIS_train_2828.csv"), dtype='float'),
            np.loadtxt(os.path.join(PKG_ROOT, ARDIS_PATH,
                                    "ARDIS_train_labels.csv"),
                       dtype='float')), \
           (np.loadtxt(os.path.join(PKG_ROOT, ARDIS_PATH,
                                    "ARDIS_test_2828.csv"),
                       dtype='float'),
            np.loadtxt(os.path.join(PKG_ROOT, ARDIS_PATH,
                                    "ARDIS_test_labels.csv"),
                       dtype='float'))


def prepare_data(x_train, x_test):
    return (x_train / 255).reshape(-1, 28, 28, 1), \
           (x_test / 255).reshape(-1, 28, 28, 1)
