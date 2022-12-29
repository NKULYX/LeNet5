# coding=utf-8
import numpy as np
import struct
import os
import random


def load_mnist(file_dir, is_images=True):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    if is_images:
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    if is_images:
        mat_data = np.reshape(mat_data, [num_images, num_rows, num_cols])
    else:
        mat_data = np.reshape(mat_data, [num_images])
    return mat_data


def get_mnist_data():
    mnist_dir = "data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    train_images = load_mnist(os.path.join(mnist_dir, train_data_dir), True)
    train_labels = load_mnist(os.path.join(mnist_dir, train_label_dir), False)
    test_images = load_mnist(os.path.join(mnist_dir, test_data_dir), True)
    test_labels = load_mnist(os.path.join(mnist_dir, test_label_dir), False)

    train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2)))
    test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)))

    _, H, W = train_images.shape
    train_images = train_images.astype(np.float32).reshape(-1, 1, H, W)
    test_images = test_images.astype(np.float32).reshape(-1, 1, H, W)

    validation_images = train_images[-1000:]
    validation_labels = train_labels[-1000:]
    train_images = train_images[:-1000]
    train_labels = train_labels[:-1000]

    mean_image = np.mean(train_images, axis=0)
    train_images -= mean_image
    validation_images -= mean_image
    test_images -= mean_image

    return {
        "X_train": train_images,
        "y_train": train_labels,
        "X_val": validation_images,
        "y_val": validation_labels,
        "X_test": test_images,
        "y_test": test_labels,
    }


def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

