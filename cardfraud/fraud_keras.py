# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    csv_file = '/home/fra/DataMart/keras/datasets/creditcard.csv'
    df = pd.read_csv(csv_file)
    # transform data
    df.pop('Time')
    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001
    df['Log Ammount'] = np.log(df.pop('Amount')+eps)
    return df

def split_df(df, target_label, test_prop=0.2):
    # Use a utility from sklearn to split and shuffle your dataset.
    train_df, test_df = train_test_split(df, test_size=test_prop)
    train_df, val_df = train_test_split(train_df, test_size=test_prop)

    # Form np arrays of labels and features.
    y_train = np.array(train_df.pop(target_label))
    y_val = np.array(val_df.pop(target_label))
    y_test = np.array(test_df.pop(target_label))

    # boolean mask
    bool_train_labels = y_train != 0

    X_train = np.array(train_df)
    X_val = np.array(val_df)
    X_test = np.array(test_df)

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def transform(X_train, X_val, X_test):
    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # remove outliers
    X_train = np.clip(X_train, -5, 5)
    X_val = np.clip(X_val, -5, 5)
    X_test = np.clip(X_test, -5, 5)

    return (X_train, X_val, X_test)

def build_model(num_features):
    # initial bias setting at the last layer
    model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(num_features,)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    df = load_data()
    (X_train, y_train, X_val, y_val, X_test, y_test) = split_df(df, 'Class')
    (X_train, X_val, X_test) = transform(X_train, X_val, X_test)
    model = build_model(X_train.shape[-1])

    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

    # params
    EPOCHS = 100
    BATCH_SIZE = 2048

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(X_val, y_val))