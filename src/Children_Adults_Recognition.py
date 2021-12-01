import numpy as np
import cv2 as cv
import os

Children_test = "Image/Children_test"
Children_train = "Image/Children_train"
Adults_test = "Image/Adults_test"
Adults_train = "Image/Adults_train"


def read_img_batch(path, endpoint=None):
    container = []
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            container.append((cv.imread(path, cv.IMREAD_GRAYSCALE)))
    return container


# read image from each group
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
x_train = np.array(x_children_train + x_adults_train)
y_train = np.append(y_children_train, y_adults_train)
x_test = np.array(x_children_test + x_adults_test)
y_test = np.append(y_children_test, y_adults_test)

print(len(x_train))
print(len(x_test))
#np.stack()

# print(y_children_train)
# print(y_adult_train)