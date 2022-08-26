import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import math

def get_dataframe():

    def readucr(filename):
        "split feature / label and trian / test"
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        # transform target to classes 0 or 1
        f = lambda x: 0 if -x == -1 else 1
        vf = np.vectorize(f)
        y = vf(y)
        return x, y.astype(int)

    data_folder = '/home/fra/DataMart/datacentre/opendata/time_series/FordA/'
    train_filename = 'FordA_TRAIN.tsv'
    test_filename = 'FordA_TEST.tsv'

    x_train, y_train = readucr(data_folder + train_filename)
    x_test, y_test = readucr(data_folder + test_filename)

    return (x_train, y_train, x_test, y_test)

def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def make_single_output_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    # one dimensional output: must use sigmoid not softmax
    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
    # output_layer = keras.layers.Dense(1, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def train_model(model):

    epochs = 5
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "cnn_best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    best_model = keras.models.load_model("cnn_best_model.h5")
    test_loss, test_acc = best_model.evaluate(x_test, y_test)

    # predict and evaluate auc for a single class
    probas = best_model.predict(x_test)
    score = roc_auc_score(y_test, probas[:,1])
    print(score)

def test_single_output_model(model):
    epochs = 100
    batch_size = 32

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam", loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
        # optimizer="adam", loss="sparse_categorical_crossentropy",
        # metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

if __name__ == '__main__':
    (x_train, y_train, x_test, y_test) = get_dataframe()

    # expand dim??
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    # print(x_train.shape)
    # (3601, 500, 1)
    # print(y_train.shape)
    # (3601,)

    # model = make_model(input_shape=x_train.shape[1:], num_classes=2)
    # train_model(model)

    model = make_single_output_model(input_shape=x_train.shape[1:], num_classes=1)
    test_single_output_model(model)
    # AUC: 0.96