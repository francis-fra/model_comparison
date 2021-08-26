from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# Model 1
def test_model01():
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

    mse_test = model.evaluate(X_test, y_test)
    # 0.427

# ------------------------------------------------------------
# Model 2
def test_model02():
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    # take another input (wide) path
    concat = keras.layers.concatenate([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[input_], outputs=[output])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

    mse_test = model.evaluate(X_test, y_test)
    # 0.418

# ------------------------------------------------------------
# Model 3
def test_model03():
    # different feature inputs
    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]

    # out of sample testing
    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
    # different??
    # X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

    # model structure
    input_A = keras.layers.Input(shape=[5], name="wide_input")

    # feature 2-7 (deep)
    input_B = keras.layers.Input(shape=[6], name="deep_input")
    hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

    concat = keras.layers.concatenate([input_A, hidden2])
    output = keras.layers.Dense(1, name="output")(concat)

    model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))

    history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))

    mse_test = model.evaluate((X_test_A, X_test_B), y_test)
    # 0.42
    
def build_model_structure(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    "return a compiled model"
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


def test_model04():
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model_structure)
    keras_reg.fit(X_train, y_train, epochs=20,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    mse_test = keras_reg.score(X_test, y_test)
    # 0.3875

from scipy.stats import reciprocal
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def test_model05():
    "hyperparameter tuning"

    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model_structure)

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
    rnd_search_cv.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    print("CV score")
    rnd_search_cv.score(X_test, y_test)

    model = rnd_search_cv.best_estimator_.model
    print("out of smaple score")
    model.evaluate(X_test, y_test)
    # 0.3346

# ------------------------------------------------------------
# test_model02()
# test_model03()
# test_model04()
test_model05()