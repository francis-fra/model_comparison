import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def load_data(test_prop=0.2):
    csv_file = '/home/fra/DataMart/keras/datasets/petfinder-mini/petfinder-mini.csv'
    df = pd.read_csv(csv_file)
    df['target'] = np.where(df['AdoptionSpeed']==4, 0, 1)
    df = df.drop(columns=['AdoptionSpeed', 'Description'])
    # train / test
    train, test = train_test_split(df, test_size=test_prop)
    # train / validation
    train, val = train_test_split(train, test_size=test_prop)
    return (train, val, test)

# A utility method to create a tf.data dataset from a Pandas df
def df_to_dataset(df, target_name, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop(target_name)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

# ------------------------------------------------------------
# custom preprocessing layers
# ------------------------------------------------------------
def get_normalization_layer(name, dataset):
    """
        PARAMETERS:
        --------
        normalize and extract a column
        name    : string column name
        dataset : tf dataset

        RETURNS:
        --------
        tf preprocessing layer
    """
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization(axis=None)
    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)
    return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    "produce a categorical encoding function (accepting a tensor)"
    if dtype == 'string':
        # string to indices mapping
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        # map integer to capped  range of indices 
        index = preprocessing.IntegerLookup(max_tokens=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)
    # One hot encoding: Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))

# TODO: simplify based on data type
def preprocess(ds):
    # list of columns and list of preprocessing layers
    all_inputs = []
    encoded_features = []

    # Numeric features.
    for header in ['PhotoAmt', 'Fee']:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # Categorical features encoded as integers.
    age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
    encoding_layer = get_category_encoding_layer('Age', ds, dtype='int64', max_tokens=5)
    encoded_age_col = encoding_layer(age_col)
    all_inputs.append(age_col)
    encoded_features.append(encoded_age_col)

    # categorical features.
    categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, ds, dtype='string', max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    return (all_inputs, encoded_features)

def build_model(all_features, encoded_features):
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    return model

if __name__ == '__main__':
    # params
    batch_size = 256
    epochs = 20

    (train_df, val_df, test_df) = load_data()
    train_ds = df_to_dataset(train_df, target_name='target', batch_size=batch_size)

    val_ds = df_to_dataset(val_df, target_name='target', shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test_df, target_name='target', shuffle=False, batch_size=batch_size)

    # get preprocessing layers
    (all_inputs, encoded_features) = preprocess(train_ds)

    model = build_model(all_inputs, encoded_features)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=["accuracy"])

    model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)