import pandas as pd
import numpy as np
from numpy import save
from numpy import load
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
import random
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image, ImageChops, ImageEnhance
from IPython.display import display  # to display images
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import roc_auc_score


# https://gist.github.com/cirocosta/33c758ad77e6e6531392
# error level analysis of an image
def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""

    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)

    except:

        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()

    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # print(d[x, y])
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff


def main():
    """
    docstring
    """
    x_casia = load('X_casia.npy')

    y_casia = load('Y_casia.npy')
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        x_casia, y_casia, test_size=0.2, random_state=5)
    
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                     activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    #optimizer = Adam()
    
    optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=2,
                                   verbose=2, mode='auto')
    
    history = model.fit(X_train, Y_train, batch_size=100, epochs=30,
                        validation_data=(X_val, Y_val), verbose=1, callbacks=[early_stopping])
    

    model.save('new_model_casia2.h5')
    

if __name__ == '__main__':
    main()
