from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from scikitplot.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


def train_lr(x, y):
    lr_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000).fit(x, y)
    n_feature = x.shape[1]
    theta = []
    for i in range(0, n_feature + 1):
        if i == 0:
            theta.append(lr_model.intercept_[0])
        else:
            theta.append(lr_model.coef_[0][i - 1])
    return (lr_model, theta)


def read_img_batch(path, endpoint=None):
    cwd = os.getcwd().replace('\\', '/')
    container = []
    for filename in os.listdir('{}/src/{}'.format(cwd, path)):
        container.append((cv.imread('{}/src/{}/{}'.format(cwd, path, filename), cv.IMREAD_GRAYSCALE)))
    return container


def convert_to_vector(imgs):
    ret = []
    for img in imgs:
        ret.append(img.reshape(img.size))
    return np.array(ret)


if __name__ == '__main__':
    Children_test = "Image/Children_test"
    Children_train = "Image/Children_train"
    Adults_test = "Image/Adults_test"
    Adults_train = "Image/Adults_train"
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

    model, _ = train_lr(convert_to_vector(x_train), convert_to_vector(y_train))

    pred = model.predict(convert_to_vector(x_test))
    accuracy = (pred == y_test).astype(int).sum()/pred.size
    cm_lr = confusion_matrix(y_test, model.predict(convert_to_vector(x_test)))
    tn, fp, fn, tp = cm_lr.ravel()
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print("accuracy: {}".format((tn + tp) / (tn + tp + fn + fp)))

    plot_confusion_matrix(y_test, pred)

    fpr, tpr, _ = roc_curve(
        y_test, model.decision_function(convert_to_vector(x_test)))
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.show()
