import idx2numpy 
import os
import numpy as np

def load_mnist():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # root project
    DATA_DIR = os.path.join(BASE_DIR, "data")

    train_images_path = os.path.join(DATA_DIR, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(DATA_DIR, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(DATA_DIR, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte')

    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    return train_images, train_labels, test_images, test_labels





