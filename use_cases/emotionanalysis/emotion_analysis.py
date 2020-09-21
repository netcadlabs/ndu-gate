import time
from threading import Thread
from random import choice
from string import ascii_lowercase

from cv2 import cv2
from numpy import expand_dims, np
# from tensorflow.python.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import img_to_array

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.dedectors.face_dedector import FaceDedector
from ndu_gate_camera.dedectors.person_dedector import PersonDedector
from use_cases.emotionanalysis.emotional_model import loadModel


class EmotionAnalysisRunner(Thread, NDUCameraRunner):

    def __init__(self, camera_service, config, connector_type):
        super().__init__()
        self.setName(config.get("name", 'EmotionAnalysisRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        self.__personDedector = PersonDedector()
        self.__faceDedector = FaceDedector()

        self.__emotion_model = loadModel()
        self.__emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def get_name(self):
        return "EmotionAnalysisRunner"

    def get_settings(self):
        settings = {'interval': 2, 'always': True}
        return settings

    def process_frame(self, frame):
        super().process_frame(frame)

        pedestrian_boxes, num_pedestrians, imageList = self.__personDedector.find_person(frame)

        result = {}

        if len(imageList) > 0:
            personCounter = 0
            faceList = self.__faceDedector.face_detector3(frame, num_pedestrians)
            for face in faceList:
                res = self.getEmotionsAnalysis2(face)
                # TODO - test et
                log.debug(res)

        return result

    def getEmotionsAnalysis2(self, img):
        emotions_sum = [0, 0, 0, 0, 0, 0, 0]
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img_pixels = img_to_array(img)
            img_pixels = expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            emotion_predictions = self.__emotion_model.predict(img_pixels)[0, :]
            sum_of_predictions = emotion_predictions.sum()
            for i in range(0, len(self.__emotion_labels)):
                emotion_label = self.__emotion_labels[i]
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

            color = (255, 0, 0)
            emotions_sum[np.argmax(emotion_predictions)] += 1
            return self.__emotion_labels[np.argmax(emotion_predictions)] + " " + str(np.argmax(emotion_predictions)), color
        except Exception as e:
            print("EMOTION ERROR...  ")
            print(e)
            emotions_sum[6] += 1
            color = (255, 0, 255)
            return "Error", color
