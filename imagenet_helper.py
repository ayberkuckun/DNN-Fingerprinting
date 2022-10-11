import os

import numpy as np


def unpickle(file):
    dict = np.load(file, allow_pickle=True)
    return dict


def load_databatch(data_folder, idx, label100, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx) + '.npz')
    x = d['data']
    y = d['labels']

    label_indexes = np.in1d(y, label100)
    y = y[label_indexes]
    x = x[label_indexes]

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = y - 1

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))

    # create mirrored images
    # data_size = x.shape[0]
    # X_train = x[0:data_size, :, :, :]
    # Y_train = y[0:data_size]
    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return x, y
