# reference for tfds
# https://medium.com/@nutanbhogendrasharma/tensorflow-image-classification-with-tf-flowers-dataset-e36205deb8fc
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras

from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def transform_sentence(s):
    return s.numpy().decode('UTF-8')
def transform_label(s):
    return s.numpy()
def split_data_label(trainds):
    return [ (transform_sentence(item['sentence']), transform_label(item['label'])) for item in trainds]
def extract_list_tuple(lst, idx):
    return [item[idx] for item in lst]

def get_data(input_length):
    [trainds, testds], info = tfds.load('glue/sst2', split=['train[:70%]', 'train[:70%]'],
                                    shuffle_files=True, with_info=True)

    # training_data = split_data_label(trainds)
    # validation_data = split_data_label(testds)
    training_data = split_data_label(trainds.take(10000))
    validation_data = split_data_label(testds.take(10000))

    training_sentences = extract_list_tuple(training_data, 0)
    training_labels = np.array(extract_list_tuple(training_data, 1))

    testing_sentences = extract_list_tuple(validation_data, 0)
    testing_labels = np.array(extract_list_tuple(validation_data, 1))

    (tokenizer, training_padded, testing_padded) = tokenize_and_pad(training_sentences, 
                                                    testing_sentences, input_length)

    return (training_padded, training_labels, testing_padded, testing_labels)


def tokenize_and_pad(training_sentences, testing_sentences, input_length):
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

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

if __name__ == '__main__':

    # params
    # batch_size = 32
    # num distinct words to keep in word dict
    vocab_size = 1000
    # input dim: length of input (integer seq) to the embedding layer
    input_length = 20
    # output_dim: output of embedding layers is 2d with sape (input_length, embedding dim)
    # i.e. for each input inetger, it is converted into a 16-element vector
    embedding_dim = 16

    (training_padded, training_labels, testing_padded, testing_labels) = get_data(input_length) 

    model = build_model()
    # 86%

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    num_epochs = 10
    model.fit(training_padded, training_labels, epochs=num_epochs, 
                validation_data=(testing_padded, testing_labels))