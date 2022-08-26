import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten, Dense

imdb_dir = '/home/fra/DataMart/datacentre/text_data/aclImdb'

def get_raw_data(folder, train_or_test):

    # folder: source folder
    # train_or_test: string: 'train' or 'test'

    src_dir = os.path.join(folder, train_or_test)

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(src_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    
    return (texts, labels)

# TODO: https://keras.io/examples/nlp/text_classification_from_scratch/

# def one_hot_encoding(texts, max_features):
#     tokenizer = Tokenizer(num_words=max_features)
#     tokenizer.fit_on_texts(texts)
#     results = tokenizer.texts_to_matrix(texts, mode='binary')
#     return (tokenizer, results)

def build_model(max_features, out_dimension, maxlen):
    """
        max_features  : max embedding layer vocab size
        out_dimension : layer out dimension
        maxlen        : vector input size
    """
    model = Sequential()
    model.add(Embedding(max_features, out_dimension, input_length=maxlen))
    # After the Embedding layer, our activations have shape `(samples, maxlen, 8)`.

    # We flatten the 3D tensor of embeddings into a 2D tensor of shape `(samples, maxlen * 8)`
    model.add(Flatten())

    # We add the classifier on top
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_transfer_model(max_features, out_dimension, maxlen):
    """
        max_features  : max embedding layer vocab size
        out_dimension : layer out dimension
        maxlen        : vector input size
    """
    model = Sequential()
    model.add(Embedding(max_features, out_dimension, input_length=maxlen))
    # FIXME: add flatten() before dense??
    # After the Embedding layer, our activations have shape `(samples, maxlen, 8)`.
    model.add(Dense(32, activation='relu'))
    # We flatten the 3D tensor of embeddings into a 2D tensor of shape `(samples, maxlen * 8)`
    model.add(Flatten())

    # We add the classifier on top
    model.add(Dense(1, activation='sigmoid'))
    return model


def test01():

    # max num of unique words in the embedding layers
    max_features = 100000
    # fixed input vector length
    input_dimension = 20
    # embedding layer output dimesnion
    out_dimension = 8

    # preprocessing
    (train_texts, labels) = get_raw_data(imdb_dir, 'train')
    tokenizer = Tokenizer(num_words=max_features)
    # This builds the word index
    tokenizer.fit_on_texts(train_texts)
    # transform words into integers (indices)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    # transform into fixed length
    x_train = preprocessing.sequence.pad_sequences(train_sequences, maxlen=input_dimension)
    y_train = np.array(labels)

    # model training
    # model = build_model(input_dimension, max_features, out_dimension)
    model = build_model(max_features, out_dimension, input_dimension)
    print(model.summary())

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # out of sample testing
    (test_texts, test_labels) = get_raw_data(imdb_dir, 'test')
    y_test = np.array(test_labels)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_test = preprocessing.sequence.pad_sequences(test_sequences, maxlen=input_dimension)
    model.evaluate(x_test, y_test, verbose=1)

def get_glove_embeddings():
    "load glove vectors"
    glove_dir = '/home/fra/DataMart/datacentre'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def shrink_embedding_matrix(embeddings_index, max_words):
    """
        embeddings_index  : word vector dictionary
        max_words         : max number of vector to keep
    """
    # vector dimension of the word vectors
    embedding_dim = list(embeddings_index.values())[0].shape[0]

    embedding_matrix = np.zeros((max_words, embedding_dim))
    # discard vectors after reaching the limit
    keys = list(embeddings_index.keys())
    for idx, word in enumerate(keys):
        embedding_vector = embeddings_index.get(word)
        if idx < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[idx] = embedding_vector
        else:
            break
    return embedding_matrix


# FIXME: even worse than model 1!?
def test02():
    # max num of vocabulary (unique words) in the embedding layers
    max_features = 100000
    # fixed input (train/test) vector length
    input_dimension = 20
    # embedding layer output dimension (must be the same as the word embedding vector size)
    out_dimension = 100

    # preprocessing
    (train_texts, labels) = get_raw_data(imdb_dir, 'train')
    tokenizer = Tokenizer(num_words=max_features)
    # This builds the word index
    tokenizer.fit_on_texts(train_texts)
    # transform words into integers (indices)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    # transform into fixed length
    x_train = preprocessing.sequence.pad_sequences(train_sequences, maxlen=input_dimension)
    y_train = np.array(labels)

    # get glove encodings
    embedding_matrix = get_glove_embeddings()
    embedding_matrix = shrink_embedding_matrix(embedding_matrix, max_features)
    assert(embedding_matrix.shape[1] == out_dimension)
    
    # transfer learning
    model = build_transfer_model(max_features, out_dimension, input_dimension)
    print(model.summary())
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2)

    # out of sample testing
    (test_texts, test_labels) = get_raw_data(imdb_dir, 'test')
    y_test = np.array(test_labels)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_test = preprocessing.sequence.pad_sequences(test_sequences, maxlen=input_dimension)
    model.evaluate(x_test, y_test, verbose=1)

if __name__ == '__main__':
    # test01()
    test02()