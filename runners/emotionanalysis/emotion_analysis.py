from threading import Thread
from random import choice
from string import ascii_lowercase
import os
import zipfile
from pathlib import Path

from ndu_gate_camera.utility.ndu_utility import NDUUtility

try:
    import tensorflow as tf
except ImportError:
    if NDUUtility.install_package("tensorflow") == 0:
        import tensorflow as tf

import cv2
import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log


class EmotionAnalysisRunner(Thread, NDUCameraRunner):

    def __init__(self, config, connector_type):
        super().__init__()
        self.setName(config.get("name", 'EmotionAnalysisRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        # self.__personDetector = PersonDetector()
        # self.__faceDetector = FaceDetector()

        self.__emotion_model = self.load_emotion_model()
        self.__emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def get_name(self):
        return "EmotionAnalysisRunner"

    def get_settings(self):
        settings = {'interval': 2, 'always': True, 'person': True, 'face': True}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)

        # pedestrian_boxes, num_pedestrians, image_list = self.__personDetector.find_person(frame)

        num_pedestrians = extra_data.get("num_pedestrians", 0)
        person_image_list = extra_data.get("person_image_list", 0)
        face_list = extra_data.get("face_list", None)

        result = []

        if len(person_image_list) > 0:
            person_counter = 0
            # face_list = self.__faceDetector.face_detector3(frame, num_pedestrians)
            for face in face_list:
                res = self._get_emotions_analysis(face)
                log.debug(res)
                result.append(res)

        return result

    def _get_emotions_analysis(self, img):
        emotions_sum = [0, 0, 0, 0, 0, 0, 0]
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))

            img_pixels = tf.keras.preprocessing.image.img_to_array(img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            emotion_predictions = self.__emotion_model.predict(img_pixels)[0, :]
            sum_of_predictions = emotion_predictions.sum()
            for i in range(0, len(self.__emotion_labels)):
                emotion_label = self.__emotion_labels[i]
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

            color = (255, 0, 0)
            emotions_sum[np.argmax(emotion_predictions)] += 1
            res_label_predict = self.__emotion_labels[np.argmax(emotion_predictions)] + " " + str(np.argmax(emotion_predictions))
            return res_label_predict, color
        except Exception as e:
            log.error(e)
            emotions_sum[6] += 1
            color = (255, 0, 255)
            return "Error", color



    try:
        import tensorflow as tf
    except ImportError:
        if NDUUtility.install_package("tensorflow") == 0:
            import tensorflow as tf


    def load_emotion_model(self):
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

        local_facial_weights_file = os.path.dirname(os.path.abspath(__file__)) + "/data/deepface/facial_expression_model_weights.h5".replace("/", os.path.sep)

        facial_weights_file = os.path.abspath(os.path.abspath(__file__) + "/data/deepface/facial_expression_model_weights.h5".replace("/", os.path.sep))

        facial_weights_file_final = None

        if os.path.isfile(local_facial_weights_file) is True:
            facial_weights_file_final = local_facial_weights_file
        elif os.path.isfile(facial_weights_file) is False:
            home = str(Path.home())
            facial_weights_file = home + ('/.deepface/weights/facial_expression_model_weights.h5').replace("/", os.path.sep)
            if os.path.isfile(facial_weights_file) is False:
                print("facial_expression_model_weights.h5 will be downloaded...")

                # zip
                url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
                output = facial_weights_file

                from gdown import download as download_g
                download_g(url, output, quiet=False)

                # unzip facial_expression_model_weights.zip
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(home + '/.deepface/weights/')

            facial_weights_file_final = facial_weights_file

        model.load_weights(facial_weights_file_final)

        return model