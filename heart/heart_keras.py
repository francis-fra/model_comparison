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

def convert2dataset(df):
    val_dataframe = df.sample(frac=0.2)
    train_dataframe = df.drop(val_dataframe.index)
    # dataset
    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)
    return train_ds, val_ds


def dataframe_to_dataset(dataframe, target_col="target"):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

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

def get_input_layers():
    # Categorical features encoded as integers
    sex = keras.Input(shape=(1,), name="sex", dtype="int64")
    cp = keras.Input(shape=(1,), name="cp", dtype="int64")
    fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
    restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
    exang = keras.Input(shape=(1,), name="exang", dtype="int64")
    ca = keras.Input(shape=(1,), name="ca", dtype="int64")
    # Categorical feature encoded as string
    thal = keras.Input(shape=(1,), name="thal", dtype="string")
    # Numerical features
    age = keras.Input(shape=(1,), name="age")
    trestbps = keras.Input(shape=(1,), name="trestbps")
    chol = keras.Input(shape=(1,), name="chol")
    thalach = keras.Input(shape=(1,), name="thalach")
    oldpeak = keras.Input(shape=(1,), name="oldpeak")
    slope = keras.Input(shape=(1,), name="slope")
    all_inputs = [ sex, cp, fbs, restecg, exang, ca, thal, age, trestbps, chol, thalach, oldpeak, slope ]
    return all_inputs

def get_encoded_input_layers(all_inputs, train_ds):
    [sex, cp, fbs, restecg, exang, ca, thal, age, trestbps, chol, thalach, oldpeak, slope ] = all_inputs
    # Integer categorical features
    sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
    cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
    fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
    restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
    exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
    ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)
    # String categorical features
    thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)
    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", train_ds)
    trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
    chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
    thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
    oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
    slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

    all_features = layers.concatenate( [
        sex_encoded,
        cp_encoded,
        fbs_encoded,
        restecg_encoded,
        exang_encoded,
        slope_encoded,
        ca_encoded,
        thal_encoded,
        age_encoded,
        trestbps_encoded,
        chol_encoded,
        thalach_encoded,
        oldpeak_encoded,
    ])
    return all_features

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

def evaluate_model(model, ds):
    y = get_label_from_batch_dataset(ds)
    probas = model.predict(ds)
    score = roc_auc_score(y, probas)
    pred = [1 if item > 0.5 else 0 for item in probas]
    acc = accuracy_score(y, pred)
    print(f"auc = {score:.4}, acc={acc:.4}")

def evaluate_XYmodel(model, X, y):
    probas = model.predict(X)
    score = roc_auc_score(y, probas)
    pred = [1 if item > 0.5 else 0 for item in probas]
    acc = accuracy_score(y, pred)
    print(f"auc = {score:.4}, acc={acc:.4}")


def test_model_01(df):
    train_ds, val_ds = convert2dataset(df)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    all_inputs = get_input_layers()
    all_features = get_encoded_input_layers(all_inputs, train_ds)

    model = get_model(all_inputs, all_features)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=100, validation_data=val_ds)
    evaluate_model(model, val_ds)

def test_model_02(df):
    target = hp.get_single_column(df, 'target')
    pipeline = Pipeline([
                        ('Var Dropper', hp.VarDropper(excl=['target'])),
                        ('Imputer', hp.Imputer()),
                        ('Encoder', hp.Encoder())])
    df = pipeline.transform(df)
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.3)
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_val = scaler.transform(X_test)

    model = keras.Sequential(
        [
            keras.layers.Dense( 64, activation="relu", input_shape=(X_train.shape[-1],)),
            keras.layers.Dense(32, activation="relu"),
            #keras.layers.Dropout(0.3),
            keras.layers.Dense(8, activation="relu"),
            #keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        Xs_train,
        y_train,
        batch_size=100,
        epochs=100,
        validation_data=(Xs_val, y_test),
    )
    evaluate_XYmodel(model, Xs_val, y_test)


if __name__ == '__main__':
    df = load_data()
    test_model_01(df)
    # auc = 0.9532, acc=0.9016
    # test_model_02(df)
    # auc = 0.8483, acc=0.7802