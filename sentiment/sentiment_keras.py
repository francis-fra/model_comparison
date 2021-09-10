# udacity tensorflow chapter 8
# https://keras.io/api/layers/core_layers/embedding/
# vocab_size = 5000
# batch_size = 50
# input_length = 10
# embedding_dim = 64

# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
# input_array = np.random.randint(5, size=(batch_size, input_length))
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)

# input shape: (batch_size, input_length)
# output shape: (batch_size, input_length, embedding_dim)

import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras
from keras import layers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# num distinct words to keep in word dict
vocab_size = 1000
# input dim: length of input (integer seq) to the embedding layer
input_length = 100
# output_dim: output of embedding layers is 2d with sape (input_length, embedding dim)
# i.e. for each input inetger, it is converted into a 16-element vector
embedding_dim = 16

def get_data(train_prop=0.8):
    words_location = '/home/fra/DataMart/datacentre/opendata/sentiment.csv'
    dataset = pd.read_csv(words_location)
    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * train_prop)

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Make labels into numpy arrays for use with the network later
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    return (training_sentences, testing_sentences, training_labels_final, testing_labels_final)

def tokenize_and_pad(train_sentences, test_sentences):
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # sequences are padded to fixed length(i.e. input_length)

    # train
    sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(sequences,maxlen=input_length, padding=padding_type, 
                        truncating=trunc_type)

    # test
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=input_length, 
                                padding=padding_type, truncating=trunc_type)

    return (tokenizer, training_padded, testing_padded)

def build_test():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    # no flatten (still works with wrong shape!)
    # model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def build_avgPooling_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    # alternative to Flatten
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Do not train...
# must be wrapped with bidirectional
def build_lstm_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(layers.LSTM(embedding_dim))
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def build_bidir_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)), 
        tf.keras.layers.Dense(6, activation='relu'), 
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_multi_bidir_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)), 
        tf.keras.layers.Dense(6, activation='relu'), 
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gru_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(layers.Bidirectional(layers.GRU(embedding_dim)))
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def build_cnn_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

if __name__ == '__main__':

    (training_sentences, testing_sentences, training_labels, testing_labels) = get_data()
    (tokenizer, training_padded, testing_padded) = tokenize_and_pad(training_sentences, testing_sentences)

    # model = build_test()
    # 43%!!
    # model = build_model()
    # 76%
    # model = build_avgPooling_model()
    # 78%
    # model = build_bidir_lstm_model()
    # 76%
    # model = build_multi_bidir_lstm_model()
    # 78%
    # model = build_lstm_model()
    # 41%
    # model = build_gru_model()
    # 80%
    model = build_cnn_model()
    # 78%

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    num_epochs = 30
    model.fit(training_padded, training_labels, epochs=num_epochs, 
            validation_data=(testing_padded, testing_labels))