from ndu_gate_camera.dedectors.model.network_model import NetworkModel


class PersonDedector:
    def __init__(self):
        self.__personDNN = NetworkModel()

    def find_person(self, frame):
        pedestrian_boxes, num_pedestrians, imageList = self.__personDNN.detect_pedestrians(frame)

        return pedestrian_boxes, num_pedestrians, imageList
