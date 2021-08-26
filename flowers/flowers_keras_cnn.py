import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub

def get_data_generators(BATCH_SIZE=32, IMG_SHAPE=150):
    data_dir = '/home/fra/DataMart/keras/datasets/flower_photos'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # ------------------------------------------------------------ 
    # train generator
    # image augmentation
    image_gen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

    train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(IMG_SHAPE,IMG_SHAPE),
                                                        class_mode='sparse')

    # ------------------------------------------------------------ 
    # validation generator
    image_gen_val = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

    val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                        directory=val_dir,
                                                        shuffle=True,
                                                        target_size=(IMG_SHAPE,IMG_SHAPE),
                                                        class_mode='sparse')

    return train_data_gen, val_data_gen


# CNN
def build_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

def test01(epochs=3):
    train_data_gen, val_data_gen = get_data_generators()
    model = build_cnn()


    checkpt = 'flowers.keras'
    # early stopping / save models
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="accuracy",
            patience=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpt,
            monitor="val_loss",
            save_best_only=True,
        )
    ]

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(
        train_data_gen,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data=val_data_gen,
    )

def test02(epochs=3):
    # required by the mobilenet
    IMG_SHAPE = 224
    train_data_gen, val_data_gen = get_data_generators(IMG_SHAPE=IMG_SHAPE)

    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(IMG_SHAPE, IMG_SHAPE,3))
    feature_extractor.trainable = False

    model = tf.keras.Sequential([
        feature_extractor,
        layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
    )

    # save model
    export_path = 'flowers_mobilenet_keras.h5'
    tf.saved_model.save(model, export_path)

def test03(epochs=3):
    "incremental learning"
    train_data_gen, val_data_gen = get_data_generators()
    model = build_cnn()

    checkpt = 'flowers.keras'
    model.load_weights(checkpt)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    history = model.fit_generator(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
    )

if __name__ == '__main__':
    # test01()
    test02(10)
    # test03()

