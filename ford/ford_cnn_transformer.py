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

# instead of 2 dim out layers, use 1 dim output
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
    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape, num_classes, head_size, num_heads, ff_dim, \
                    num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# def test_transformer_encoder_model(model):
#     epochs = 100
#     batch_size = 32

#     model.compile(
#         loss="sparse_categorical_crossentropy",
#         optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#         metrics=["sparse_categorical_accuracy"],
#     )
#     model.summary()

#     callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

#     model.fit(
#         x_train,
#         y_train,
#         validation_split=0.2,
#         epochs=200,
#         batch_size=64,
#         callbacks=callbacks,
#     )

#     model.evaluate(x_test, y_test, verbose=1)

def train_model(model):

    epochs = 100
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

    # method 1
    # model = make_model(input_shape=x_train.shape[1:], num_classes=2)
    # train_model(model)

    # method 2
    # model = make_single_output_model(input_shape=x_train.shape[1:], num_classes=1)
    # test_single_output_model(model)

    # method 3: slow
    model = build_transformer_model(
        input_shape=x_train.shape[1:],
        num_classes=2,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )
    train_model(model)