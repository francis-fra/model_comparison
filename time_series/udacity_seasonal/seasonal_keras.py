from seasonal_python import get_data, split_train_test
import tensorflow as tf

keras = tf.keras

# note that the data is 2d (i.e. (batch=32, window_size=30)) and the target is 1d (i.e. [32])
def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    "convert numpy into keras batched dataset"
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    # tuple of (data, label)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def train_linear_model(train_set, valid_set, window_size):
    "linear DL model"

    model = keras.models.Sequential([
        keras.layers.Dense(1, input_shape=[window_size])
    ])

    optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping])    

def train_dense_model(train_set, valid_set, window_size):
    "Two layered Dense model"

    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    early_stopping = keras.callbacks.EarlyStopping(patience=10)

    model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping]) 

# returns only a vector per batch
def train_seq2vec_model(train_set, valid_set, window_size):
    "sequence to vector RNN"

    model = keras.models.Sequential([
        # RNN needs inputs to be 3 dimensional: (batch_size=32, window_size=30, series dimension=1)
        # add extra dimension as data input has only 2 dimensions
        # input shape is None means any length of batches
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        keras.layers.SimpleRNN(100, return_sequences=True),
        keras.layers.SimpleRNN(100),
        keras.layers.Dense(1),
        # to help training!!
        keras.layers.Lambda(lambda x: x * 200.0)
    ])

    optimizer = keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])

    early_stopping = keras.callbacks.EarlyStopping(patience=50)
    # model_checkpoint = keras.callbacks.ModelCheckpoint( "my_checkpoint", save_best_only=True)
    model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping])

# TORM
def seq2seq_window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    "same as window dataset except the extra dimension"
    # add extra dimension here
    series = tf.expand_dims(series, axis=-1)
    return window_dataset(series, window_size, batch_size, shuffle_buffer)
    # ds = tf.data.Dataset.from_tensor_slices(series)
    # ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    # ds = ds.shuffle(shuffle_buffer)
    # ds = ds.map(lambda w: (w[:-1], w[1:]))
    # return ds.batch(batch_size).prefetch(1)

def train_seq2seq_model(train_set, valid_set, window_size):
    model = keras.models.Sequential([
        # make it 3 dimensional
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[None, 1]),
        # also return a seq
        keras.layers.SimpleRNN(100, return_sequences=True),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: x * 200.0)
    ])

    optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])

    early_stopping = keras.callbacks.EarlyStopping(patience=10)

    model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping])

# for stateful RNN
class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states() 

# for stateful RNN
# no shuffle and no overlapping windows
def sequential_window_dataset(series, window_size):
    # debug: added at the model instead
    # series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    # shift by window size, hence non-overlapping
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    # no more shuffle
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)

def train_stateful_rnn_model(train_set, valid_set, window_size):

    model = keras.models.Sequential([
        # for stateful RNN, need to know the batch size
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), batch_input_shape=[1, None]),
        keras.layers.SimpleRNN(100, return_sequences=True, stateful=True),
        keras.layers.SimpleRNN(100, return_sequences=True, stateful=True),
        keras.layers.Dense(1),
        # essential...
        keras.layers.Lambda(lambda x: x * 200.0)
    ])
    optimizer = keras.optimizers.SGD(lr=1e-7, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])

    reset_states = ResetStatesCallback()
    # model_checkpoint = keras.callbacks.ModelCheckpoint( "my_checkpoint.h5", save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=50)

    model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping, reset_states])

def train_lstm_model(train_set, valid_set, window_size):

    model = keras.models.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), batch_input_shape=[1, None]),
        # keras.layers.LSTM(100, return_sequences=True, stateful=True, batch_input_shape=[1, None, 1]),
        keras.layers.LSTM(100, return_sequences=True, stateful=True),
        keras.layers.LSTM(100, return_sequences=True, stateful=True),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: x * 200.0)
    ])

    optimizer = keras.optimizers.SGD(lr=5e-7, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    reset_states = ResetStatesCallback()

    # model_checkpoint = keras.callbacks.ModelCheckpoint( "my_checkpoint.h5", save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=50)
    model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping, reset_states])

def train_cnn_model(train_set, valid_set, window_size):
    # window_size = 64
    model = keras.models.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), batch_input_shape=[1, None]),
        keras.layers.Conv1D(filters=32, kernel_size=5,
                            strides=1, padding="causal",
                            activation="relu"),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: x * 200)
    ])

    optimizer = keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    history = model.fit(train_set, epochs=100)

# best and fastest
def train_multiple_cnn_model(train_set, valid_set, window_size):

    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), batch_input_shape=[1, None]))
    # model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for dilation_rate in (1, 2, 4, 8, 16, 32):
        model.add(
            keras.layers.Conv1D(filters=32,
                                kernel_size=2,
                                strides=1,
                                dilation_rate=dilation_rate,
                                padding="causal",
                                activation="relu")
        )

    # last layer - not dense?
    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
    optimizer = keras.optimizers.Adam(lr=3e-4)
    model.compile(loss=keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    history = model.fit(train_set, epochs=100)


# ------------------------------------------------------------
# test procedures
# ------------------------------------------------------------
def test_seq2vec():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)

    window_size = 30
    # seq to vector data set
    train_set = window_dataset(x_train, window_size)
    valid_set = window_dataset(x_valid, window_size)

    # three different models
    train_linear_model(train_set, valid_set, window_size)
    train_dense_model(train_set, valid_set, window_size)
    train_seq2vec_model(train_set, valid_set, window_size)

def test_seq2seq():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)

    window_size = 30

    # not needed...
    # seq to seq data set
    # train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
    # valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)

    # debug
    train_set = window_dataset(x_train, window_size)
    valid_set = window_dataset(x_valid, window_size)

    train_seq2seq_model(train_set, valid_set, window_size)

def test_stateful_rnn():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)

    window_size = 30

    # train_set = window_dataset(x_train, window_size)
    # valid_set = window_dataset(x_valid, window_size)
    train_set = sequential_window_dataset(x_train, window_size)
    valid_set = sequential_window_dataset(x_valid, window_size)

    train_stateful_rnn_model(train_set, valid_set, window_size)

def test_lstm():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)

    window_size = 30

    # non-overlapping windows 
    train_set = sequential_window_dataset(x_train, window_size)
    valid_set = sequential_window_dataset(x_valid, window_size)

    train_lstm_model(train_set, valid_set, window_size)

def test_cnn():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)

    # window_size =64 
    window_size = 30

    # non-overlapping windows 
    train_set = sequential_window_dataset(x_train, window_size)
    valid_set = sequential_window_dataset(x_valid, window_size)

    # train_cnn_model(train_set, valid_set, window_size)
    train_multiple_cnn_model(train_set, valid_set, window_size)

if __name__ == '__main__':
    # test_seq2vec()
    # test_seq2seq()
    # test_stateful_rnn()
    # test_lstm()
    test_cnn()