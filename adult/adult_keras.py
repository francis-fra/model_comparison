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
import kerasModel as km

def load_data(batch_size):
    "product train/test dataset and metadata"

    data_location = '/home/fra/DataMart/datacentre/opendata/UCI/adult/'
    train_data = pd.read_csv(data_location + 'adult_train_data.csv')
    test_data = pd.read_csv(data_location + 'adult_test_data.csv')
    metadata = {}

    # target has to be integer!!!
    TARGET_FEATURE_NAME = "income"
    # note the space!!
    TARGET_FEATURE_LABELS = [" <=50K", " >50K"]

    NUMERIC_FEATURE_NAMES = [
        "age",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    # part of categorical 
    INTEGER_FEATURE_NAMES = []
    # A dictionary of the categorical features and their vocabulary.
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "workclass": sorted(list(train_data["workclass"].unique())),
        "education": sorted(list(train_data["education"].unique())),
        "marital_status": sorted(list(train_data["marital_status"].unique())),
        "occupation": sorted(list(train_data["occupation"].unique())),
        "relationship": sorted(list(train_data["relationship"].unique())),
        "race": sorted(list(train_data["race"].unique())),
        "gender": sorted(list(train_data["gender"].unique())),
        "native_country": sorted(list(train_data["native_country"].unique())),
    }

    # categorical is a union set of integer and string types
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

    # exclude target
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    # all columns
    CSV_HEADER = list(train_data.columns)
    # NUM_CLASSES = len(TARGET_FEATURE_LABELS)

    metadata = kd.MetaData(TARGET_FEATURE_NAME,
                            CSV_HEADER,
                            NUMERIC_FEATURE_NAMES,
                            INTEGER_FEATURE_NAMES,
                            CATEGORICAL_FEATURE_NAMES,
                            FEATURE_NAMES,
                            CATEGORICAL_FEATURES_WITH_VOCABULARY,
                            TARGET_FEATURE_LABELS)

    train_ds = kd.df2dataset(train_data, metadata).batch(batch_size)
    val_ds = kd.df2dataset(test_data, metadata, shuffle=False).batch(batch_size)

    # encode target: encode in batch, other wise very slow!!!
    (train_ds, lookup) = kd.encode_target(train_ds, TARGET_FEATURE_LABELS)
    (val_ds, _) = kd.encode_target(val_ds, TARGET_FEATURE_LABELS, lookup)

    return (train_ds, val_ds, metadata)

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

# AUC metrics is applicable to dim 1 output only
def test_model_performance(model, train_ds, val_ds):
    print(model.summary())
    model.compile("adam", "binary_crossentropy", metrics=["accuracy", "AUC"])
    model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)
    # single column only!!
    # out = model.predict(val_ds)
    # print(out)
    model.evaluate(val_ds)


# ouput layer has dim=2??
def test_tree_performance(model, params, train_ds, val_ds):
    learning_rate = params['learning_rate']
    # only binary cross entropy can use AUC
    # errors
    # model.compile("adam", "binary_crossentropy", metrics=["accuracy", "AUC"])
    # ok
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # ok but poor performance ?
    # model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    # ERRORS
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy", "AUC"])
    model.fit(train_ds, epochs=num_epochs)
    model.evaluate(val_ds)
    # manually calculate AUC
    probas = model.predict(val_ds)
    y = kd.get_label_from_batch_dataset(val_ds)
    score = roc_auc_score(y, probas[:,1])
    print(score)

def create_tree_model(metadata):
    num_trees = 10
    depth = 10
    used_features_rate = 1.0
    num_classes = len(metadata.target_labels)

    inputs = kd.create_model_inputs(metadata)
    encoded_inputs = kd.encode_inputs(inputs, metadata)
    encoded_inputs = layers.BatchNormalization()(encoded_inputs)
    num_features = encoded_inputs.shape[1]

    tree = km.NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
    outputs = tree(encoded_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_forest_model(metadata):
    num_trees = 10
    depth = 10
    used_features_rate = 1.0
    num_classes = len(metadata.target_labels)

    inputs = kd.create_model_inputs(metadata)
    encoded_inputs = kd.encode_inputs(inputs, metadata)
    encoded_inputs = layers.BatchNormalization()(encoded_inputs)
    num_features = encoded_inputs.shape[1]

    tree = km.NeuralDecisionForest(num_trees, depth, num_features, used_features_rate, num_classes)
    outputs = tree(encoded_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    params = {
        'learning_rate' : 0.01,
        'dropout_rate' : 0.5,
        'batch_size' : 265,
        'num_epochs' : 10,
        'encoding_size' : 16,
        # 'use_embedding' : False
        'use_embedding' : True
    }
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    train_ds, val_ds, metadata = load_data(batch_size)

    # model = build_model(metadata, params, train_ds)
    # model = build_test_model(metadata, params, train_ds)
    # test_model_performance(model, train_ds, val_ds)
    
    # tree_model = create_tree_model(metadata)
    tree_model = create_forest_model(metadata)
    test_tree_performance(tree_model, params, train_ds, val_ds)
    
