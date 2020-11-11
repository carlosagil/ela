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


def train():
    """
    docstring
    """

    path_original = 'casia2/Au/'
    path_tampered = 'casia2/Tp/'

    total_original = os.listdir(path_original)
    total_tampered = os.listdir(path_tampered)

    print(f'===> Total original images {len(total_original)}')
    print(f'===> Total tampered images {len(total_tampered)}')

    images = []

    for file in tqdm(os.listdir(path_original)):
        try:
            if file.endswith('jpg'):
                if int(os.stat(path_original + file).st_size) > 10000:
                    line = path_original + file + ',0\n'
                    images.append(line)
        except:
            print(path_original+file)

    for file in tqdm(os.listdir(path_tampered)):
        try:
            if file.endswith('jpg'):
                if int(os.stat(path_tampered + file).st_size) > 10000:
                    line = path_tampered + file + ',1\n'
                    images.append(line)
            if file.endswith('tif'):
                if int(os.stat(path_tampered + file).st_size) > 10000:
                    line = path_tampered + file + ',1\n'
                    images.append(line)
        except:
            print(path_tampered+file)

    print(f'===> Total images {len(images)}')

    image_name = []
    label = []
    for i in tqdm(range(len(images))):
        image_name.append(images[i][0:-3])
        label.append(images[i][-2])

    dataset = pd.DataFrame({'image': image_name, 'class_label': label})

    dataset.head()

    dataset.to_csv('dataset_casia2.csv', index=False)

    dataset = pd.read_csv('dataset_casia2.csv')

    x_casia = []
    y_casia = []

    for index, row in tqdm(dataset.iterrows()):
        print(f'\n Analyzing {index} of {len(dataset)}')
        x_casia.append(
            np.array(ELA(row[0]).resize((128, 128))).flatten() / 255.0)
        y_casia.append(row[1])

    x_casia = np.array(x_casia)
    y_casia = np.array(y_casia)

    x_casia = x_casia.reshape(-1, 128, 128, 3)

    y_casia = to_categorical(y_casia, 2)  # y is one hot encoded

    # save all the data
    save('X_casia.npy', x_casia)
    save('Y_casia.npy', y_casia)

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

    optimizer = Adam()

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=2,
                                   verbose=1, mode='auto')

    history = model.fit(X_train, Y_train, batch_size=100, epochs=30,
                        validation_data=(X_val, Y_val), verbose=1, callbacks=[early_stopping])

    model.save('new_model_casia.h5')

    fig = plt.figure()
    p1 = fig.add_subplot(221)
    p2 = fig.add_subplot(222)
    p3 = fig.add_subplot(223)
    p4 = fig.add_subplot(224)
    p2.set_ylim(0, 1)
    p4.set_ylim(0, 1)
    p1.grid()
    p2.grid()
    p3.grid()
    p4.grid()
    p2.set_yticks(np.arange(0, 1, 0.1))
    p4.set_yticks(np.arange(0, 1, 0.1))
    x = [i for i in range(5)]
    y = history.history['loss']
    y2 = history.history['acc']
    y3 = history.history['val_loss']
    y4 = history.history['val_accuracy']
    p1.plot(x, y, 'r', label='train_loss')
    p1.legend()
    p2.plot(x, y2, 'b', label='train_accuracy')
    p2.legend()
    p3.plot(x, y3, 'r', label='val_loss')
    p3.legend()
    p4.plot(x, y4, 'b', label='val_accuracy')
    p4.legend()
    plt.show()

    y_pred_cnn = model.predict(X_val)
    y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val, axis=1)

    score = metrics.precision_score(Y_true, y_pred_cnn, average="weighted")
    print("Precision score: {}".format(score))
    score = metrics.recall_score(Y_true, y_pred_cnn, average="weighted")
    print("Recall score: {}".format(score))
    score_lr1 = metrics.f1_score(Y_true, y_pred_cnn, average="weighted")
    print("F1 score: {}".format(score_lr1))

    cm = confusion_matrix(Y_true, y_pred_cnn)
    print('Confusion matrix:\n', cm)

    print(classification_report(Y_true, y_pred_cnn))

    print('Plot of Confusion Matrix')
    df_cm = pd.DataFrame(cm, columns=np.unique(
        Y_true), index=np.unique(Y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,
                annot_kws={"size": 16})  # font size

    print('ROC_AUC score:', roc_auc_score(Y_true, y_pred_cnn))
    df_cm


if __name__ == '__main__':
    train()
