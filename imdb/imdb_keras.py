from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, LSTM, GRU 

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

def load_and_transform_data(max_features, maxlen):
    """
        max_features: max vocab (words)
        maxlen      : encoded vector length
    """
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    return (x_train, y_train, x_test, y_test)

# def build_model():
#     model = Sequential()
#     model.add(Embedding(max_features, 32))
#     model.add(SimpleRNN(32))
#     model.add(Dense(1, activation='sigmoid'))
#     return model

# def build_lstm_model():
#     model = Sequential()
#     model.add(Embedding(max_features, 32))
#     model.add(LSTM(32))
#     model.add(Dense(1, activation='sigmoid'))
#     return model

def build_model(net):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(net)
    model.add(Dense(1, activation='sigmoid'))
    return model

def test01():
    (x_train, y_train, x_test, y_test) = load_and_transform_data(max_features, maxlen)
    model = build_model(SimpleRNN(32))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

    model.evaluate(x_test, y_test, verbose=1)

def test02():
    (x_train, y_train, x_test, y_test) = load_and_transform_data(max_features, maxlen)
    model = build_model(LSTM(32))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

    model.evaluate(x_test, y_test, verbose=1)

def test03():
    (x_train, y_train, x_test, y_test) = load_and_transform_data(max_features, maxlen)
    model = build_model(GRU(32))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

    model.evaluate(x_test, y_test, verbose=1)

if __name__ == '__main__':
    # test01()
    # test02()
    test03()