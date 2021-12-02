import numpy as np
import cv2 as cv
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys

Children_test = "Image/Children_test"
Children_train = "Image/Children_train"
Adults_test = "Image/Adults_test"
Adults_train = "Image/Adults_train"


def read_img_batch(path, endpoint=None):
    container = []
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            container.append(cv.imread(path, cv.IMREAD_GRAYSCALE))
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

def normalizeData(data):
    data = np.array(data)
    data = data.astype('float32')
    data = data / 255
    return data

# combine training set and testing set
x_train = x_children_train + x_adults_train
x_train = normalizeData(x_train)
y_train = np.append(y_children_train, y_adults_train)
x_test = x_children_test + x_adults_test
x_test = normalizeData(x_test)
y_test = np.append(y_children_test, y_adults_test)

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print(y_train.shape)

#Train model
use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("cifar.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(8,(3,3),padding='same',input_shape=(48,48,3),activation='relu'))
    model.add(Conv2D(8,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

    model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2,activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    batch_size = 128
    epochs = 20
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    model.save("cifar.model")
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()