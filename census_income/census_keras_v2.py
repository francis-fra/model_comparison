import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

from functools import partial


def load_data():
    CSV_HEADER = [
        "age",
        "class_of_worker",
        "detailed_industry_recode",
        "detailed_occupation_recode",
        "education",
        "wage_per_hour",
        "enroll_in_edu_inst_last_wk",
        "marital_stat",
        "major_industry_code",
        "major_occupation_code",
        "race",
        "hispanic_origin",
        "sex",
        "member_of_a_labor_union",
        "reason_for_unemployment",
        "full_or_part_time_employment_stat",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "tax_filer_stat",
        "region_of_previous_residence",
        "state_of_previous_residence",
        "detailed_household_and_family_stat",
        "detailed_household_summary_in_household",
        "instance_weight",
        "migration_code-change_in_msa",
        "migration_code-change_in_reg",
        "migration_code-move_within_reg",
        "live_in_this_house_1_year_ago",
        "migration_prev_res_in_sunbelt",
        "num_persons_worked_for_employer",
        "family_members_under_18",
        "country_of_birth_father",
        "country_of_birth_mother",
        "country_of_birth_self",
        "citizenship",
        "own_business_or_self_employed",
        "fill_inc_questionnaire_for_veteran's_admin",
        "veterans_benefits",
        "weeks_worked_in_year",
        "year",
        "income_level",
    ]
    train_file = '/home/fra/DataMart/datacentre/opendata/UCI/census/census_income_train.csv'
    test_file = '/home/fra/DataMart/datacentre/opendata/UCI/census/census_income_test.csv'

    df = pd.read_csv(train_file)
    testdf = pd.read_csv(test_file)

    df.columns = CSV_HEADER
    testdf.columns = CSV_HEADER

    # split validation and training
    random_selection = np.random.rand(len(df.index)) <= 0.85
    traindf = df[random_selection]
    validdf = df[~random_selection]

    return traindf, validdf, testdf

# target transfrom
def data_transform(df):
    df["income_level"] = df["income_level"].apply(
        lambda x: 0 if x == " - 50000." else 1)
    return df

def get_datatype_dict(df):

    TARGET_FEATURE_NAME = "income_level"
    # TODO: what is Weight column name.??
    WEIGHT_COLUMN_NAME = "instance_weight"
    # Numeric feature names.
    NUMERIC_FEATURE_NAMES = [
        "age",
        "wage_per_hour",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "num_persons_worked_for_employer",
        "weeks_worked_in_year",
    ]
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        feature_name: sorted([str(value) for value in list(df[feature_name].unique())])
        for feature_name in df.columns
        if feature_name
            not in list(NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME, TARGET_FEATURE_NAME])
    }
    # All features names.
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(
        CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
    )

    datatype_dict = {}
    datatype_dict['numeric'] = NUMERIC_FEATURE_NAMES
    datatype_dict['target_col'] = TARGET_FEATURE_NAME
    datatype_dict['weight_col'] = WEIGHT_COLUMN_NAME
    datatype_dict['categorical_value_vocab'] = CATEGORICAL_FEATURES_WITH_VOCABULARY
    datatype_dict['features'] = FEATURE_NAMES
    datatype_dict['all_columns'] = list(df.columns)

    return datatype_dict


def create_model_inputs(datatype_dict):
    "Collection of inputs split into numerical and categorical"
    inputs = {}
    NUMERIC_FEATURE_NAMES = get_numeric_names(datatype_dict)
    FEATURE_NAMES = get_feature_names(datatype_dict)

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

def encode_inputs(inputs, params):
    "encode input layers"
    encoded_features = []
    encoding_size = params['encoding_size']
    CATEGORICAL_FEATURES_WITH_VOCABULARY = params['categorical_value_dict']
    # encode and append each feature one by one
    for feature_name in inputs:
        # categorical
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # get unique values
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            index = StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = index(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            embedding_ecoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=encoding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding_ecoder(value_index)
        else:
            # Project the numeric feature to encoding_size using linear transformation.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features

def process(params, features, target):
    "transform X, y data inputs"
    CATEGORICAL_FEATURES_WITH_VOCABULARY = params['categorical_value_dict']
    WEIGHT_COLUMN_NAME = params['weight_col_name']
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Cast categorical feature values to string.
            features[feature_name] = tf.cast(features[feature_name], tf.dtypes.string)
    # Get the instance weight.
    weight = features.pop(WEIGHT_COLUMN_NAME)
    return features, target, weight

def create_model(params, datatype_dict):
    # get data inputs
    inputs = create_model_inputs(datatype_dict)
    # create encoding layers
    feature_list = encode_inputs(inputs, params)
    num_features = len(feature_list)

    encoding_size = params['encoding_size']
    dropout_rate = params['dropout_rate']

    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# def get_dataset_from_csv(csv_file_path, params, datatype_dict, shuffle=False, batch_size=128):

#     CSV_HEADER = get_all_columns(datatype_dict)
#     NUMERIC_FEATURE_NAMES = get_numeric_names(datatype_dict)
#     WEIGHT_COLUMN_NAME = params['weight_col_name']
#     TARGET_FEATURE_NAME = params['target_col']
#     # Feature default values.
#     COLUMN_DEFAULTS = [
#         [0.0]
#         if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME, WEIGHT_COLUMN_NAME]
#         else ["NA"]
#         for feature_name in CSV_HEADER
#     ]

#     transform = partial(process, params)
#     dataset = tf.data.experimental.make_csv_dataset(
#         csv_file_path,
#         batch_size=batch_size,
#         column_names=CSV_HEADER,
#         column_defaults=COLUMN_DEFAULTS,
#         label_name=TARGET_FEATURE_NAME,
#         num_epochs=1,
#         header=False,
#         shuffle=shuffle,
#     ).map(transform)

#     return dataset

# ------------------------------------------------------------
# Network components
# ------------------------------------------------------------
# Gated Linear Units (GLUs) provide the flexibility to suppress input that are not relevant for a given task.
class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        # elementwise??
        return self.linear(inputs) * self.sigmoid(inputs)

# accept a single column once each time
class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        # units represent the fixed embedding size for all categorical columns
        self.units = units
        # exponential activation unit
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        # GLU
        self.gated_linear_unit = GatedLinearUnit(units)
        # the normalization happens across the axes within each example
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)
    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        # make sure the output shape is the same
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        # add the GRU output for value suppressing??
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        # units is the encoding size of the categorical values
        super(VariableSelection, self).__init__()
        # list of layers
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        # collect all encoded inputs
        v = layers.concatenate(inputs)
        # apply GRN for all inputs to produce feature weights
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        # feed each input to each individual GRN
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        # stack: add extra dimension
        x = tf.stack(x, axis=1)
        #  weighted sum output 
        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs

# ------------------------------------------------------------
def get_label_from_batch_dataset(ds):
    ys = np.array([])
    # tuple: (X, y)
    for tup in ds:
        ys = np.concatenate((ys, tup[1]), axis=0)
    return ys

def evaluate_model(model, ds):
    y = get_label_from_batch_dataset(ds)
    probas = model.predict(ds)
    score = roc_auc_score(y, probas)
    pred = [1 if item > 0.5 else 0 for item in probas]
    acc = accuracy_score(y, pred)
    print(f"auc = {score:.4}, acc={acc:.4}")

def get_feature_names(datatype_dict):
    return datatype_dict['features']

def get_all_columns(datatype_dict):
    return datatype_dict['all_columns']

def get_numeric_names(datatype_dict):
    return datatype_dict['numeric']

def get_categorical_names(datatype_dict):
    return list(datatype_dict['categorical_value_vocab'].keys())

def dataframe_to_dataset(dataframe, target_col):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    # ds.map(process)
    # ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def convert_categorical_columns(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('object')
        df[col] = df[col].apply(lambda x: str(x))
    return df

def build_GRU_model(trainds, validds, params, datatype_dict):
    model = create_model(params, datatype_dict)
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    model.fit(
        trainds,
        epochs=num_epochs,
        validation_data=validds,
        callbacks=[early_stopping],
    )

    evaluate_model(model, validds)

def build_dense_model(trainds, validds, params, datatype_dict):
    inputs = create_model_inputs(datatype_dict)
    feature_list = encode_inputs(inputs, params)

    x = layers.concatenate(feature_list)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    model.fit(
        trainds,
        epochs=num_epochs,
        validation_data=validds,
        callbacks=[early_stopping],
    )

    evaluate_model(model, validds)
    

if __name__ == '__main__':
    traindf, validdf, testdf = load_data()

    traindf = data_transform(traindf)
    validdf = data_transform(validdf)
    testdf = data_transform(testdf)

    datatype_dict = get_datatype_dict(traindf)
    params = {
        'learning_rate' : 0.001,
        'dropout_rate' : 0.15,
        'batch_size' : 265,
        'num_epochs' : 1,
        'encoding_size' : 16,
        'target_col': datatype_dict['target_col'],
        'categorical_value_dict': datatype_dict['categorical_value_vocab'],
        'weight_col_name': datatype_dict['weight_col'],
    }
    batch_size = params['batch_size']

    # convert categorical columns
    categorical_cols = get_categorical_names(datatype_dict)
    traindf = convert_categorical_columns(traindf, categorical_cols)
    validdf = convert_categorical_columns(validdf, categorical_cols)

    # create datasets
    trainds = dataframe_to_dataset(traindf, params['target_col'])
    validds = dataframe_to_dataset(validdf, params['target_col'])

    # change categorical dtype and convert values to string
    transform = partial(process, params)
    trainds = trainds.map(transform).batch(batch_size)
    validds = validds.map(transform).batch(batch_size)

    # build_GRU_model(trainds, validds, params, datatype_dict)
    # auc = 0.9352, acc=0.9509
    build_dense_model(trainds, validds, params, datatype_dict)
    # auc = 0.9027, acc=0.9353