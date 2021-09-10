import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from tensorflow.keras.layers import StringLookup

def get_numeric_feature_names(params):
    return params['numeric_features']

def get_feature_names(params):
    return params['features_names']

def get_categorical_feature_names(params):
    return params['categorical_features']

def get_categorical_value_dict(params):
    return params['categorical_value_dict']

def get_column_names(params):
    return params['data_columns']

def get_target_column(params):
    return params['target_col']

def get_dataset_from_csv(csv_file_path, metadata, batch_size, shuffle=False):

    CSV_HEADER = metadata['data_columns']
    TARGET_FEATURE_NAME = metadata['target_col']
    NUMERIC_FEATURE_NAMES = get_numeric_feature_names(metadata)

    COLUMN_DEFAULTS = [
        [0] if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] else ["NA"]
        for feature_name in CSV_HEADER
    ]

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=True,
        shuffle=shuffle,
    )
    return dataset.cache()

def load_datasets(batch_size=265):
    data_location = '/home/fra/DataMart/datacentre/opendata/UCI/convertype/'

    data = pd.read_csv(data_location + 'train_data.csv')
    # print(data.head())

    metadata = {}
    TARGET_FEATURE_NAME = "Cover_Type"
    TARGET_FEATURE_LABELS = ["0", "1", "2", "3", "4", "5", "6"]
    NUMERIC_FEATURE_NAMES = [
        "Aspect",
        "Elevation",
        "Hillshade_3pm",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Horizontal_Distance_To_Fire_Points",
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Slope",
        "Vertical_Distance_To_Hydrology",
    ]
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "Soil_Type": list(data["Soil_Type"].unique()),
        "Wilderness_Area": list(data["Wilderness_Area"].unique()),
    }
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

    metadata['features_names'] = FEATURE_NAMES
    metadata['numeric_features'] = NUMERIC_FEATURE_NAMES
    metadata['categorical_features'] = CATEGORICAL_FEATURE_NAMES
    metadata['categorical_value_dict'] = CATEGORICAL_FEATURES_WITH_VOCABULARY
    metadata['data_columns'] = list(data.columns)
    metadata['target_col'] = TARGET_FEATURE_NAME
    metadata['num_classes'] = len(TARGET_FEATURE_LABELS)

    train_data_file = data_location + 'train_data.csv'
    test_data_file = data_location + 'test_data.csv'

    train_dataset = get_dataset_from_csv(train_data_file, metadata, batch_size, shuffle=True)
    test_dataset = get_dataset_from_csv(test_data_file, metadata, batch_size)

    return (train_dataset, test_dataset, metadata)

def create_model_inputs(metadata):
    inputs = {}
    FEATURE_NAMES = get_feature_names(metadata)
    NUMERIC_FEATURE_NAMES = get_numeric_feature_names(metadata)
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


def encode_inputs(inputs, metadata, use_embedding=False):
    encoded_features = []
    CATEGORICAL_FEATURE_NAMES = get_categorical_feature_names(metadata)
    CATEGORICAL_FEATURES_WITH_VOCABULARY = get_categorical_value_dict(metadata)
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                # Convert the string input values into integer indices.
                encoded_feature = lookup(inputs[feature_name])
                embedding_dims = int(math.sqrt(len(vocabulary)))
                # Create an embedding layer with the specified dimensions.
                embedding = layers.Embedding(
                    input_dim=len(vocabulary), output_dim=embedding_dims
                )
                # Convert the index values to embedding representations.
                encoded_feature = embedding(encoded_feature)
            else:
                # Convert the string input values into a one hot encoding.
                encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        else:
            # Use the numerical features as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)

        encoded_features.append(encoded_feature)

    all_features = layers.concatenate(encoded_features)
    return all_features

def create_baseline_model(metadata, params):
    inputs = create_model_inputs(metadata)
    features = encode_inputs(inputs, metadata)
    NUM_CLASSES = metadata['num_classes']
    dropout_rate = params['dropout_rate']

    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# baseline_model = create_baseline_model(metadata, params)
# keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

def run_experiment(model, params, train_dataset, test_dataset):

    learning_rate = params['learning_rate']
    num_epochs = params['num_epochs']

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    # test_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")

    _, accuracy = model.evaluate(test_dataset, verbose=0)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

if __name__ == '__main__':
    (train_dataset, test_dataset, metadata) = load_datasets()
    params = {
        'learning_rate' : 0.001,
        'dropout_rate' : 0.1,
        'batch_size' : 265,
        'num_epochs' : 50,
        'encoding_size' : 16,
        'target_col': get_target_column(metadata),
        'categorical_value_dict': get_categorical_value_dict(metadata),
        'num_classes': metadata['num_classes']
    }
    hidden_units = [32, 32]
    # print(metadata)
    model = create_baseline_model(metadata, params)
    run_experiment(model, params, train_dataset, test_dataset)