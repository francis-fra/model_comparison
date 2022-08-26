import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from sklearn.linear_model import LinearRegression

def load_data_fashion_mnist(batch_size):
    train_data = '/home/fra/DataMart/images/fashion-mnist/fashion-mnist_train.csv'
    test_data = '/home/fra/DataMart/images/fashion-mnist/fashion-mnist_test.csv'
    train = pd.read_csv(train_data)
    train_tensor = tf.Variable(train.values, dtype="float")
    test = pd.read_csv(test_data)
    test_tensor = tf.Variable(test.values, dtype="float")

    X_train = train_tensor[:,:-1] / 255.
    X_test = test_tensor[:,:-1] / 255.

    y_train = train_tensor[:,0]
    y_test = test_tensor[:,0]
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(y_train))
    train_iter = dataset.batch(batch_size)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_iter = dataset.batch(batch_size)
    
    return train_iter, test_iter

def cross_entropy(y_hat, y):
    """
        PARAMETERS
        ---------
        y_hat   : estimated probabilities for each class
        y       : integer, indices of labels
    """
    yy = tf.cast(y, "int32")
    x = tf.boolean_mask(y_hat, tf.one_hot(yy, depth=y_hat.shape[-1]))
    return -tf.math.log(x)

def sgd(params, grads, lr, batch_size):
    "sgd in-place param update"
    # update param, not for propagation
    for param, grad in zip(params, grads):
        param.assign( param - lr * grad / batch_size)

def net(X, W, b):
    x = tf.matmul(X, W) + b
    return softmax(x)

def softmax(X):
    X_exp = tf.math.exp(X)
    # normalizer
    partition = tf.reduce_sum(X_exp, axis=1, keepdims=True)
    return X_exp / partition 

def relu(X):
    mask = tf.cast(X > 0, "float")
    return X * mask

def manual_run(train_set, val_set, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']
    num_inputs = metadata['num_inputs']
    num_classes = metadata['num_classes']

    # single dense layer
    W = tf.Variable(initial_value=tfp.distributions.Normal(0, 0.01).sample([num_inputs, num_classes]), 
                    dtype="float", shape=(num_inputs, num_classes))
    b = tf.Variable(num_classes, dtype="float")

    # train
    for k in range(epochs):
        for X, y in train_set:
            with tf.GradientTape() as t:
                y_hat = net(X, W, b)
                l = cross_entropy(y_hat, y)
            grads = t.gradient(l, [W, b])
            sgd([W, b], grads, lr, batch_size)
        print(f'epoch {k + 1}, loss {float(tf.reduce_mean(l)):f}')

    evaluate_accuracy(net, val_set, [W, b])

def auto_run(train_set, val_set, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']
    num_inputs = metadata['num_inputs']
    num_classes = metadata['num_classes']

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # y is expected to be one-hot encoded
    # loss = tf.keras.losses.CategoricalCrossentropy()
    # y is expected to be integer
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for k in range(epochs):
        for X, y in train_set:
            with tf.GradientTape() as t:
                pred = model(X)
                l = loss(y, pred)
            grads = t.gradient(l, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'epoch {k + 1}, loss {float(tf.reduce_mean(l)):f}')

    evaluate_accuracy(model, val_set)


def keras_run(train_set, val_set, metadata):
    # batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']
    # num_inputs = metadata['num_inputs']
    num_classes = metadata['num_classes']

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # y is expected to be one-hot encoded
    # loss = tf.keras.losses.CategoricalCrossentropy()
    # y is expected to be integer
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    sgd = tf.keras.optimizers.SGD(learning_rate=lr)

    model.compile(loss=loss, optimizer = sgd)
    model.fit(train_set, epochs=epochs, validation_data=val_set)

    evaluate_accuracy(model, val_set)

def count_correct(pred, y):
    """
        pred   : tf tensor
        y      : tf tensor
    """
    yy = tf.cast(y, "int64")
    return tf.reduce_sum(tf.cast(yy == pred, "int64"))


def evaluate_accuracy(model, val_set, params=None):
    """
        model   : fitted model
        val_set : tf dataset
        params  : additional params of model
    """
    correct = 0
    total = 0
    for X, y in val_set:
        if params is not None:
            pred = model(X, *params)
        else:
            pred = model(X)
        # get hard decision
        predictions = tf.math.argmax(pred, axis=1)
        # print(predictions.dtype, y.dtype)
        # int64, float32
        correct += count_correct(predictions, y)
        total += len(y)

    print(f"accuracy: {correct / total:.2f}")

if __name__ == '__main__':
    metadata = {
        'epochs': 10,
        'lr' : 1e-2,
        'batch_size': 256, 
        'num_classes': 10,
        'num_inputs': 784
    } 
    train_set, val_set = load_data_fashion_mnist(batch_size=metadata['batch_size'])
    # manual_run(train_set, val_set,metadata)
    # auto_run(train_set, val_set,metadata)
    keras_run(train_set, val_set,metadata)