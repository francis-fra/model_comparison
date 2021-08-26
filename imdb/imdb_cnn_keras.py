from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from keras.models import Sequential
from tensorflow.keras import layers 

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


def build_model(max_features, max_len):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    return model

def test01():
    max_features = 10000
    maxlen = 500

    (x_train, y_train, x_test, y_test) = load_and_transform_data(max_features, maxlen)
    model = build_model()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

    model.evaluate(x_test, y_test, verbose=1)

if __name__ == '__main__':
    test01()