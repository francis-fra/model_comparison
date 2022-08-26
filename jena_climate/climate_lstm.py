import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

def load_data(normalize=True):
    data_dir = '/home/fra/DataMart/datacentre/opendata/'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    # feature column names
    titles = [
        "Pressure",
        "Temperature",
        "Temperature in Kelvin",
        "Temperature (dew point)",
        "Relative Humidity",
        "Saturation vapor pressure",
        "Vapor pressure",
        "Vapor pressure deficit",
        "Specific humidity",
        "Water vapor concentration",
        "Airtight",
        "Wind speed",
        "Maximum wind speed",
        "Wind direction in degrees",
    ]

    with open(fname) as f:
        data = f.read()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    # normalize
    if normalize == True:
        mean = float_data[:200000].mean(axis=0)
        float_data -= mean
        std = float_data[:200000].std(axis=0)
        float_data /= std

    return (float_data, titles)

def filter_columns(data, columns):
    # change to df
    cols = [0, 1, 5, 7, 8, 10, 11]
    selected_features = [columns[i] for i in  cols]
    df = pd.DataFrame(data[:,cols], columns=selected_features)
    return df

def split_data(df, train_split):
    "split into train/test"
    train_df = df[0 : train_split]
    val_df = df[train_split:]
    return (train_df, val_df)

def calculate_sampling_params(params, test_prop):
    # first target for prediction
    params['start'] = params['past'] + params['future']
    # num training samples required
    params['sequence_length'] = int(params['past'] / params['step'])
    # proportion of train/test split
    # test_prop = 0.285
    params['train_split'] = int((1-test_prop) * int(df.shape[0]))
    return params

def make_sequence(train_df, val_df, target, params):
    # target end point (length of the training samples)
    start = params['start']
    end = start + params['train_split']
    sequence_length = int(params['past'] / params['step'])

    x_train = train_df.values
    y_train = target[start:end]

    x_end = len(val_df) - params['past'] - params['future']
    label_start = params['train_split'] + params['past'] + params['future']

    x_val = val_df.iloc[:x_end].values
    y_val = target[label_start:]

    return (x_train, y_train, x_val, y_val)

def get_dataset(df, params):
    # temperature is the target
    target = df.iloc[:,1].to_numpy().reshape(-1, 1)
    # split train / test 
    (train_df, val_df) = split_data(df, params['train_split'])
    # print(train_df.head())
    # print(train_df.shape)
    # print(val_df.shape)
    # (300622, 7)
    # (119829, 7)

    (x_train, y_train, x_val, y_val) = make_sequence(train_df, val_df, target, params)
    # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    # (300622, 7) (300622,) (119037, 7) (119037,)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train, y_train,
        sequence_length=params['sequence_length'],
        sampling_rate=params['step'],
        batch_size=params['batch_size'],
    )

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=params['sequence_length'],
        sampling_rate=params['step'],
        batch_size=params['batch_size'],
    )

    return dataset_train, dataset_val

def build_lstm_model(params):
    input_dim = params['input_dim']
    learning_rate = params['learning_rate']
    inputs = keras.layers.Input(shape=(input_dim[1], input_dim[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_rnn_model(params):
    input_dim = params['input_dim']
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(input_dim[1], input_dim[2])))
    model.add(layers.Dense(1))
    return model

def build_dense_model(params):
    input_dim = params['input_dim']
    model = Sequential()
    model.add(layers.Flatten(input_shape=(input_dim[1], input_dim[2])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    return model

def build_stacking_rnn_model(params):
    input_dim = params['input_dim']
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, 
                            recurrent_dropout=0.5, 
                            return_sequences=True,
                            input_shape=(input_dim[1], input_dim[2])))
    model.add(layers.GRU(64, activation='relu',
                            dropout=0.1, 
                            recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    return model

def train_model(model, params, dataset_train, dataset_val):
    epochs = params['epochs']
    learning_rate = params['learning_rate']

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[es_callback],
    )
class NaiveModel:
    def __init__(self):
        pass

    def predict(self, X):
        # predict based on last temperature
        return X[:, -1, 1]

def evaluate_model(model, val_ds):
    batch_maes = []
    for samples, targets in val_ds:
        preds = model.predict(samples)
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


if __name__ == '__main__':
    (data, columns) = load_data()
    # print(data.shape)
    # (420451, 14)

    df = filter_columns(data, columns)
    # print(df.shape)
    # (420451, 7)

    # using 720 sample to predict values 72 timestamp ahead
    params = {
        # lookback seq length
        'past': 720,
        # predict 72 timestamps ahead
        'future': 72,
        # sample once every 6 obs
        'step': 6,
        'learning_rate' : 0.001,
        'batch_size' : 256,
        'epochs' : 3
    }

    params = calculate_sampling_params(params, 0.285)
    dataset_train, dataset_val = get_dataset(df, params)

    # get shape 
    for batch in dataset_train.take(1):
        inputs, targets = batch
    # (batch_size, num train_hist, columns)
    # Input shape: (256, 120, 7)
    # Target shape: (256, 1)

    params['input_dim'] = inputs.numpy().shape
    params['target_dim'] = targets.numpy().shape

    # model 1
    # lstm_model = build_lstm_model(params)
    # train_model(lstm_model, params, dataset_train, dataset_val)
    # evaluate_model(lstm_model, dataset_val)

    # model 2
    # rnn_model = build_rnn_model(params)
    # train_model(rnn_model, params, dataset_train, dataset_val)
    # evaluate_model(rnn_model, dataset_val)

    # model 3
    # naive_model = NaiveModel()
    # evaluate_model(naive_model, dataset_val)

    # model 4
    dense_model = build_dense_model(params)
    train_model(dense_model, params, dataset_train, dataset_val)
    evaluate_model(dense_model, dataset_val)

    # model 5
    stack_model = build_stacking_rnn_model(params)
    train_model(stack_model, params, dataset_train, dataset_val)
    evaluate_model(stack_model, dataset_val)