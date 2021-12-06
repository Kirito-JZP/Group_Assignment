import cv2 as cv
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import ReadImage as RI
from keras.optimizers import Adam,SGD

x_train, y_train, x_test, y_test = RI.make_dataset()
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print(y_train)

#Train model
use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("CNN.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=(48, 48, 1), activation='relu'))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2,activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    opt = Adam(lr=0.0001, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    batch_size = 64
    epochs = 30
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    model.save("CNN.model")
