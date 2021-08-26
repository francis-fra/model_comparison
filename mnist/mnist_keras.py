from tensorflow import keras
# from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

# X_train_full.shape
# X_train_full[0].max()
# y_train_full[:5]

# normalize
X_train_full = X_train_full / 255.
X_test = X_test / 255.


X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)


def test_model01():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=X_train.shape[1:]),
        keras.layers.Dense(120, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    # X dimension: (batch, *image_dim)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

    model.evaluate(X_test, y_test)
    # 0.9335

def test_model02():

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu"))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    # X dimension: (batch, *image_dim)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
    model.evaluate(X_test, y_test)
    # 0.961

def train_modelA():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))

    for n_hidden in (300, 100, 50, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu"))

    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
    model.evaluate(X_test, y_test)
    return model

def test_transfer_learning():
    model_A = train_modelA()
    # transfer learning
    model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
    model_B_on_A.add(keras.layers.Dense(10, activation="softmax"))
    # for layer in model_B_on_A.layers[:-1]:
    #     layer.trainable = False
    model_B_on_A.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    history = model_B_on_A.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
    model_B_on_A.evaluate(X_test, y_test)
    # 0.9625

def test_model03():

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu"))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", 
                    optimizer=keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
                    metrics=["accuracy"])
    # X dimension: (batch, *image_dim)
    history = model.fit(X_train, y_train, epochs=5, 
                        validation_data=(X_valid, y_valid), 
                        callbacks=[lr_scheduler])
    model.evaluate(X_test, y_test)
    # 0.977

def test_model04():

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu", 
                    kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", 
                    optimizer=keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
                    metrics=["accuracy"])
    # X dimension: (batch, *image_dim)
    history = model.fit(X_train, y_train, epochs=5, 
                        validation_data=(X_valid, y_valid), 
                        callbacks=[lr_scheduler])
    model.evaluate(X_test, y_test)
    # 0.8557

def test_model05():

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu"))
        model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", 
                    optimizer=keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
                    metrics=["accuracy"])
    # X dimension: (batch, *image_dim)
    history = model.fit(X_train, y_train, epochs=5, 
                        validation_data=(X_valid, y_valid), 
                        callbacks=[lr_scheduler])
    model.evaluate(X_test, y_test)
    # 0.953

if __name__ == '__main__':
    # test_model01()
    # test_model02()
    # test_transfer_learning()
    # test_model03()
    # test_model04()
    test_model05()
