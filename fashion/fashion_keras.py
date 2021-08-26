import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

num_validation = 5000

# normalize
X_valid, X_train = X_train_full[:num_validation] / 255., X_train_full[num_validation:] / 255.
y_valid, y_train = y_train_full[:num_validation], y_train_full[num_validation:]

X_test = X_test / 255.

# y_train[:5]

# Keras Model
# Dense
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)
# acc: 0.83