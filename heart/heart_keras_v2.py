import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wpy as hp

import tensorflow as tf
keras = tf.keras

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup

def load_data():
    file_location = '/home/fra/DataMart/datacentre/opendata/UCI/heart.csv'
    df = pd.read_csv(file_location)
    return df

def dataframe_to_dataset(dataframe, target_col="target"):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def convert2dataset(df):
    val_dataframe = df.sample(frac=0.2)
    train_dataframe = df.drop(val_dataframe.index)
    # dataset
    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)
    return train_ds, val_ds

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset, is_string):
    "encode a single column"
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def get_input_layers(feature_dict):
    "data input layers"
    # list of tuples: (name, datatype, Keras.Input)
    all_inputs = []
    for col, datatype in feature_dict.items():
        if datatype == "integer":
            all_inputs.append((col, datatype, keras.Input(shape=(1,), name=col, dtype="int64")))
        elif datatype == "categorical":
            all_inputs.append((col, datatype, keras.Input(shape=(1,), name=col, dtype="string")))
        elif datatype == "numerical":
            all_inputs.append((col, datatype, keras.Input(shape=(1,), name=col)))
        else:
            raise Exception("unknown data type")
    return all_inputs

def get_encoded_input_layers(all_inputs, train_ds):
    "encoded input layers"
    encoded_inputs = []
    for colname, datatype, feature in all_inputs:
        if datatype == "integer":
            encoded_inputs.append(encode_categorical_feature(feature, colname, train_ds, False))
        elif datatype == "categorical":
            encoded_inputs.append(encode_categorical_feature(feature, colname, train_ds, True))
        elif datatype == "numerical":
            encoded_inputs.append(encode_numerical_feature(feature, colname, train_ds))
        else:
            raise Exception("unknown data type")
    return layers.concatenate(encoded_inputs)

def extract_inputs(inputs_metadata):
    return [keras_input for _, _, keras_input in inputs_metadata]

def get_model(all_inputs, all_features):
    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)
    return model

def get_label_from_batch_dataset(ds):
    ys = np.array([])
    for x, y in ds:
        ys = np.concatenate((ys, y), axis=0)
    return ys

# WARNING: do not use shuffled data set for model evaluation
def evaluate_model(model, ds):
    y = get_label_from_batch_dataset(ds)
    probas = model.predict(ds)
    score = roc_auc_score(y, probas)
    pred = [1 if item > 0.5 else 0 for item in probas]
    acc = accuracy_score(y, pred)
    print(f"auc = {score:.4}, acc={acc:.4}")

def get_datatype_dict():
    integer_columns = ["sex", "cp", "fbs", "restecg", "exang", "ca"]
    string_columns = ["thal"]
    numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    datatype_dict = {}
    for col in integer_columns:
        datatype_dict[col] = "integer"
    for col in string_columns:
        datatype_dict[col] = "categorical"
    for col in numeric_columns:
        datatype_dict[col] = "numerical"
    return datatype_dict


def test_model_01(df):
    train_ds, val_ds = convert2dataset(df)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    datatype_dict = get_datatype_dict()

    inputs_metadata = get_input_layers(datatype_dict)
    all_inputs = extract_inputs(inputs_metadata)
    all_features = get_encoded_input_layers(inputs_metadata, train_ds)

    model = get_model(all_inputs, all_features)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=100, validation_data=val_ds)
    evaluate_model(model, val_ds)

# Much Much better???
if __name__ == '__main__':
    df = load_data()
    test_model_01(df)
    # auc = 0.9238, acc=0.8361