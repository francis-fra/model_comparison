import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

def load_data(normalize=True):
    data_dir = '/home/fra/DataMart/datacentre/opendata/'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

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

    return float_data

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    """
        data      : raw data
        lookback  : historical lookback
        delay     : prediction look ahead time 
        min_index : time index (lower)
        max_index : time index (upper)
        shuffle   : random selection
        batch_size: num samples per batch
        step      : num data points to draw per hours

    """
    
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    # generator loop
    while 1:
        if shuffle:
            # random sampling
            rows = np.random.randint( min_index + lookback, max_index, size=batch_size)
        else:
            # sequential
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        
        # fork into data and label
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            # index 1: temperature
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

def evaluate_model(model, gen, steps):
    "predict based on last "
    batch_maes = []
    for step in range(steps):
        samples, targets = next(gen)
        preds = model.predict(samples)
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

class NaiveModel:
    def __init__(self):
        pass

    def predict(self, X):
        # predict based on last temperature
        return X[:, -1, 1]

def get_dense_model(dim):
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, dim)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    return model

def get_rnn_model(dim):
    model = Sequential()
    # model.add(layers.GRU(32, input_shape=(None, dim)))
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, dim)))
    model.add(layers.Dense(1))
    return model

def get_stacking_rnn_model(dim):
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, 
                            recurrent_dropout=0.5, 
                            return_sequences=True,
                            input_shape=(None, dim)))
    model.add(layers.GRU(64, activation='relu',
                        dropout=0.1, 
                        recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    return model

def get_generators(data, lookback, step, delay, batch_size):
    train_gen = generator(data,
                      lookback=lookback, delay=delay, min_index=0,
                      max_index=200000, shuffle=True, step=step, batch_size=batch_size)
    val_gen = generator(data, lookback=lookback, delay=delay,
                        min_index=200001, max_index=300000, step=step, batch_size=batch_size)
    test_gen = generator(data, lookback=lookback, delay=delay,
                        min_index=300001, max_index=None, step=step, batch_size=batch_size)


    # This is how many steps to draw from `val_gen` in order to see the whole validation set
    val_steps = (300000 - 200001 - lookback) // batch_size
    # This is how many steps to draw from `test_gen` in order to see the whole test set
    test_steps = (len(data) - 300001 - lookback) // batch_size

    return (train_gen, val_gen, test_gen, val_steps, test_steps)

def test_naive_model():
    # params
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    # get data sets
    data = load_data()   
    dim = data.shape[-1]

    (train_gen, val_gen, test_gen, val_steps, test_steps) = get_generators(data, lookback, step, delay, batch_size)

    # naive model performance
    naive_model = NaiveModel()
    evaluate_model(naive_model, test_gen, test_steps)
    # 0.3056

def test_dense_model():
    # params
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    # get data sets
    data = load_data()   
    dim = data.shape[-1]

    (train_gen, val_gen, test_gen, val_steps, test_steps) = get_generators(data, lookback, step, delay, batch_size)

    model = get_dense_model(dim)
    model.compile(optimizer=RMSprop(), loss='mae')

    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=10,
                        validation_data=val_gen,
                        validation_steps=val_steps)

    evaluate_model(model, test_gen, test_steps)
    # 0.438

def test_rnn_model():
    # params
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    # get data sets
    data = load_data()   
    dim = data.shape[-1]

    (train_gen, val_gen, test_gen, val_steps, test_steps) = get_generators(data, lookback, step, delay, batch_size)

    # two different structures
    # model = get_rnn_model(dim)
    model = get_stacking_rnn_model(dim)
    model.compile(optimizer=RMSprop(), loss='mae')

    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=10,
                        validation_data=val_gen,
                        validation_steps=val_steps)

    evaluate_model(model, test_gen, test_steps)
    # 0.421

def get_cnn_rnn_model(dim):
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, dim)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    return model

def test_cnn_rnn_model():
    # params
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    # get data sets
    data = load_data()   
    dim = data.shape[-1]

    (train_gen, val_gen, test_gen, val_steps, test_steps) = get_generators(data, lookback, step, delay, batch_size)

    model = get_cnn_rnn_model(dim)
    model.compile(optimizer=RMSprop(), loss='mae')

    history = model.fit(train_gen,
                        steps_per_epoch=500,
                        epochs=10,
                        validation_data=val_gen,
                        validation_steps=val_steps)

    evaluate_model(model, test_gen, test_steps)
    # 0.432


if __name__ == '__main__':
    # test_naive_model()
    # test_dense_model()
    # test_rnn_model()
    test_cnn_rnn_model()
