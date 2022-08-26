import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def load_data():
    location = '/home/fra/DataMart/datacentre/opendata/mtcars.csv'
    df = pd.read_csv(location)
    target_col = 'mpg'
    # feature_col = ['hp', 'wt']
    feature_col = ['hp', 'wt', 'cyl']
    cols = feature_col + [target_col]
    out = df[cols].astype('float32')
    # out = df[cols]
    return out

def data_iter(batch_size, X, y):
    """create an iterator to yield batched tensors
        PARAMETERS 
        ----------
        X, y: torch tensors

        RETURN
        ------
        batched X, y
    
    """
    num_examples = len(X)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield X[batch_indices], y[batch_indices]

def split_and_standardize(df, target_col):
    "standardize feature columns"
    y = df[target_col].values
    y = y.reshape(-1, 1)
    X = df.drop(target_col, axis=1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def linear_regression(X, y):
    "sklearn regression"
    reg = LinearRegression().fit(X, y)
    print(f"R2 score: {reg.score(X, y)}")
    print(f"coeff: {reg.coef_}")
    print(f"intercept: {reg.intercept_}")

def tf_regression(X, y):
    metadata = {
        'epochs': 300,
        'lr' : 1e-1,
        'batch_size': 16 
    } 
    # manual_run(X, y, metadata)
    auto_run(X, y, metadata)
    # keras_run(X, y, metadata)

def linreg(X, w, b):
    """The linear regression model."""
    return tf.matmul(X, w) + b
    # return tf.tensordot(X, w, axes=1) + b

def loss(y_hat, y):
    """MSE loss"""
    yy = tf.reshape(y, y_hat.shape)
    return (y_hat - yy)**2 / 2

def sgd(params, grads, lr, batch_size):
    """optimizer"""
    for param, grad in zip(params, grads):
        param.assign(param - lr * grad / batch_size)

def manual_run(X, y, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']

    features = tf.convert_to_tensor(X, dtype="float")
    label = tf.convert_to_tensor(y, dtype="float")
    num_features = list(features.shape)[1]

    W = tf.Variable(initial_value=tfp.distributions.Normal(0, 0.01).sample([num_features, 1]), 
                    dtype="float", shape=(num_features,1))
    b = tf.Variable(0, dtype="float")

    for k in range(epochs):
        for X, y in data_iter(batch_size, X, y):
            with tf.GradientTape() as t:
                y_hat = linreg(X, W, b)
                l = loss(y_hat, y)
                grads = t.gradient(l, [W, b])
            sgd([W, b], grads, lr, batch_size)
        # print(f'epoch {k + 1}, loss {float(tf.math.reduce_mean(l)):f}')

    # evaluate
    y_pred = linreg(features, W, b)
    score = r2_score(label, y_pred)
    print(f"R2 score: {score}")
    print(f"coeff: {W}")
    print(f"intercept: {b}")

def create_batched_dataset(batch_size, features, labels, shuffle=True):
    """
    PARAMETERS
    ----------
    batch_size      : integer 
    features, labels: tf tensors

    RETURN
    ------
    batched data set
    """
    data_arrays = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataset))
    return dataset.batch(batch_size)


def keras_run(X, y, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']

    features = tf.convert_to_tensor(X, dtype="float")
    labels = tf.convert_to_tensor(y, dtype="float")
    dataset = create_batched_dataset(batch_size, features, labels, shuffle=True)

    initializer = tf.initializers.RandomNormal(stddev=0.01)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

    loss = tf.keras.losses.MeanSquaredError()
    sgd = tf.keras.optimizers.SGD(learning_rate=lr)

    r2 = tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
    model.compile(optimizer=sgd, loss=loss, metrics=[r2, "mean_squared_error"])
    model.fit(dataset, epochs=epochs)

    # evaluate
    y_pred = model(features)
    score = r2_score(labels, y_pred)
    print(f"R2 score: {score}")
    print(f"coeff: {model.trainable_variables}")


def auto_run(X, y, metadata):
    batch_size = metadata['batch_size']
    epochs = metadata['epochs']
    lr = metadata['lr']

    features = tf.convert_to_tensor(X, dtype="float")
    label = tf.convert_to_tensor(y, dtype="float")
    num_features = list(features.shape)[1]

    initializer = tf.initializers.RandomNormal(stddev=0.01)
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    dataset = create_batched_dataset(batch_size, features, label)

    for k in range(epochs):
        # for X, y in data_iter(batch_size, X, y):
        for X, y in dataset:
            with tf.GradientTape() as t:
                y_hat = net(X)
                l = loss(y_hat, y)
            grads = t.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

    # evaluate
    y_pred = net(features)
    score = r2_score(label, y_pred)
    print(f"R2 score: {score}")
    print(f"coeff: {net.trainable_variables}")

if __name__ == '__main__':
    df = load_data()
    X, y = split_and_standardize(df, 'mpg')
    linear_regression(X, y)
    tf_regression(X, y)