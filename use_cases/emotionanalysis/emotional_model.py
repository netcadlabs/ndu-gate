import os
from pathlib import Path
from sys import path

import tensorflow as tf;

# from keras.models import Model, Sequential
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

import zipfile


def loadModel():
    num_classes = 7

    model = tf.keras.models.Sequential()

    # 1st convolution layer
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    # fully connected neural networks
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # ----------------------------

    current_dir = path.dirname(path.dirname(path.abspath(__file__)))

    facial_expression_model_weights = path.abspath(current_dir + "/data/deepface/facial_expression_model_weights.h5")

    if os.path.isfile(facial_expression_model_weights) is False:

        home = str(Path.home())

        facial_expression_model_weights = home + '/.deepface/weights/facial_expression_model_weights.h5'
        if os.path.isfile(facial_expression_model_weights) is False:
            print("facial_expression_model_weights.h5 will be downloaded...")

            # TO-DO: upload weights to google drive

            # zip
            url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
            output = facial_expression_model_weights

            from gdown import download as download_g
            download_g(url, output, quiet=False)

            # unzip facial_expression_model_weights.zip
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(home + '/.deepface/weights/')

    model.load_weights(facial_expression_model_weights)

    return model

    # ----------------------------

    return 0
