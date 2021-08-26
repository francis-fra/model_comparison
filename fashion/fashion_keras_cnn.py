import tensorflow as tf
import numpy as np
from tensorflow import keras
from functools import partial

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

num_validation = 5000

# normalize
X_valid, X_train = X_train_full[:num_validation] / 255., X_train_full[num_validation:] / 255.
y_valid, y_train = y_train_full[:num_validation], y_train_full[num_validation:]

X_test = X_test / 255.

# extend the third dimension
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train2 = X_train.reshape(X_train.shape[0], 28*28, 1)
X_valid2 = X_valid.reshape(X_valid.shape[0], 28*28, 1)
X_test2 = X_test.reshape(X_test.shape[0], 28*28, 1)

def test01():


    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential([
        DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))
    score = model.evaluate(X_test, y_test)

def test02():


    # inputs need to be 4 dim with batch
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential([
        DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))
    score = model.evaluate(X_test, y_test)

def test03():
    # input reshaped needed

    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=7, input_shape=[28*28, 1]),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=128, kernel_size=7),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=256, kernel_size=7),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = model.fit(X_train2, y_train, epochs=2, validation_data=(X_valid2, y_valid))
    score = model.evaluate(X_test2, y_test)


if __name__ == '__main__':
    # test01()
    # test02()
    test03()