import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

lib_location = '/home/fra/Project/pyProj/zqlib/zq/dl'
sys.path.append(lib_location)
import kerasData as kd

def load_data():
    "product train/test dataset and metadata"

    data_location = '/home/fra/DataMart/datacentre/opendata/UCI/'
    data = pd.read_csv(data_location + 'heart.csv')
    metadata = {}

    TARGET_FEATURE_NAME = "target"
    TARGET_FEATURE_LABELS = ["0", "1"]

    NUMERIC_FEATURE_NAMES = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
    # part of categorical (optional??)
    INTEGER_FEATURE_NAMES = ["sex", "cp", "fbs", "restecg", "exang", "ca"]

    # categorical also includes interger feature
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        feature_name: sorted([str(value) for value in list(data[feature_name].unique())])
        for feature_name in data.columns
        if feature_name
            not in list(NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME])
    }
    # categorical is a union set of integer and string types
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

    # exclude target
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    # all columns
    CSV_HEADER = list(data.columns)
    # NUM_CLASSES = len(TARGET_FEATURE_LABELS)

    metadata = kd.MetaData(TARGET_FEATURE_NAME,
                            CSV_HEADER,
                            NUMERIC_FEATURE_NAMES,
                            INTEGER_FEATURE_NAMES,
                            CATEGORICAL_FEATURE_NAMES,
                            FEATURE_NAMES,
                            CATEGORICAL_FEATURES_WITH_VOCABULARY,
                            TARGET_FEATURE_LABELS)
                            # NUM_CLASSES)

    train_ds, val_ds = kd.convert2dataset(data, metadata)

    return (train_ds, val_ds, metadata)

def build_model(metadata, params, train_ds):
    dropout_rate = params['dropout_rate']

    inputs = kd.create_model_inputs(metadata)
    encoded_inputs = kd.encode_inputs(inputs, metadata)

    x = encoded_inputs
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_test_model(metadata, params, train_ds):
    dropout_rate = params['dropout_rate']
    use_embedding = params['use_embedding']

    # dictionary of input layers
    inputs = kd.create_normalized_model_inputs(metadata, use_embedding)
    encoded_inputs = kd.encode_and_normalize_inputs(inputs, metadata, params, train_ds, use_embedding)

    x = encoded_inputs
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_normalized_model(metadata, params, ds):

    dropout_rate = params['dropout_rate']

    # prepare and normalize inputs
    inputs_metadata = kd.get_input_layers(metadata)
    encoded_inputs = kd.get_encoded_input_layers(inputs_metadata, metadata, ds)

    # functional API
    x = encoded_inputs
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    inputs = kd.extract_inputs(inputs_metadata)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def test_model_performance(model, train_ds, val_ds):
    print(model.summary())
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)
    kd.evaluate_model(model, val_ds)

def test_model_performance_with_cb(model, train_ds, val_ds):

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    model.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=val_ds,
        callbacks=[early_stopping],
    )

    kd.evaluate_model(model, val_ds)


# normalization is very important!!
# early stopping is also important
if __name__ == '__main__':
    params = {
        'learning_rate' : 0.01,
        'dropout_rate' : 0.5,
        'batch_size' : 32,
        'num_epochs' : 50,
        'encoding_size' : 16,
        'use_embedding' : False
        # 'use_embedding' : True
    }
    batch_size = params['batch_size']

    train_ds, val_ds, metadata = load_data()
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']

    # model 1: embedding option: Y; normalization: N (as in census model)
    # model01 = build_model(metadata, params, train_ds)
    # test_model_performance(model01, train_ds, val_ds)

    # model 2: embedding option: Y; normalization: Y
    model02 = build_test_model(metadata, params, train_ds)
    test_model_performance(model02, train_ds, val_ds)
    # test_model_performance_with_cb(model02, train_ds, val_ds)

    # model 3: embedding option: N; normalization: Y (as in heart model)
    # model03 = build_normalized_model(metadata, params, train_ds)
    # test_model_performance(model03, train_ds, val_ds)

