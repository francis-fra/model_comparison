import pickle
import numpy as np
import tensorflow as tf
keras = tf.keras

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow_datasets as tfds

from tensorflow.keras.applications import vgg19


def format_image(image, label, image_size=(224, 224)):
    image = tf.image.resize(image, image_size)/255.0
    return image, label

def load_data(batch_size=32):
    (training_set, validation_set), dataset_info = tfds.load( 'tf_flowers',
        split=['train[:70%]', 'train[70%:]'],
        with_info=True,
        as_supervised=True,
    )
    num_classes = dataset_info.features['label'].num_classes

    train_batches = training_set.map(format_image).batch(batch_size).prefetch(1)
    validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)

    return (train_batches, validation_batches, num_classes)


def build_vgg19model(num_classes, image_size=224):
    vgg_model = vgg19.VGG19(weights="imagenet", include_top=False, 
                            input_shape=(image_size,image_size,3))

    model = keras.Sequential([
        vgg_model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    vgg_model.trainalbe = False
    return model

if __name__ == '__main__':
    (train_batches, validation_batches, num_classes) =  load_data()
    model = build_vgg19model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc']) 
    epochs = 2
    model.fit(train_batches, epochs=epochs, validation_data=validation_batches, verbose=1)

    # save model
    export_path = 'flowers_vgg19_keras.h5'
    tf.saved_model.save(model, export_path)