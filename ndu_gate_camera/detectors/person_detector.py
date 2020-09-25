from ndu_gate_camera.detectors.model.network_model import NetworkModel


class PersonDetector:
    def __init__(self):
        self.__personDNN = NetworkModel()

    def find_person(self, frame):
        pedestrian_boxes, num_pedestrians, image_list = self.__personDNN.detect_pedestrians(frame)

        return pedestrian_boxes, num_pedestrians, image_list
