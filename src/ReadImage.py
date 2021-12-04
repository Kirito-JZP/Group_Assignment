import numpy as np
from PIL import Image
import os

Children_test = "Image/Children_test"
Children_train = "Image/Children_train"
Adults_test = "Image/Adults_test"
Adults_train = "Image/Adults_train"


def convert_to_grayscale(rgb_img):
    y_cb_cr_img = rgb_img.convert('YCbCr')
    y, cb, cr = y_cb_cr_img.split()
    return y


def read_img_batch(path, endpoint=None):
    container = []
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            pic = Image.open(path)
            pic = np.array(convert_to_grayscale(pic))
            pic = np.reshape(pic, (48, 48, 1))
            container.append(pic)
    return container


def make_dataset():
    x_children_train = read_img_batch(Children_train)
    x_children_test = read_img_batch(Children_test)
    x_adults_train = read_img_batch(Adults_train)
    x_adults_test = read_img_batch(Adults_test)

    # set label according to each image set
    # children 0; adults 1
    y_children_train = np.zeros(len(x_children_train), dtype=int)
    y_children_test = np.zeros(len(x_children_test), dtype=int)
    y_adults_train = np.ones(len(x_adults_train), dtype=int)
    y_adults_test = np.ones(len(x_adults_test), dtype=int)

    # combine training set and testing set
    x_train = x_children_train + x_adults_train
    y_train = np.append(y_children_train, y_adults_train)
    x_test = x_children_test + x_adults_test
    y_test = np.append(y_children_test, y_adults_test)
    return x_train, y_train, x_test, y_test