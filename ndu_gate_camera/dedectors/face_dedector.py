import time
from sys import path

from cv2 import cv2

from ndu_gate_camera.dedectors.vision.ssd.config.fd_config import define_img_size
from ndu_gate_camera.dedectors.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor


class FaceDedector:
    label_path = path.dirname(path.dirname(path.abspath(__file__))) + "/data/voc-model-labels.txt"
    faceModelPath = path.dirname(path.dirname(path.abspath(__file__))) + "/data/version-RFB-640.pth"

    def __init__(self, threshold=0.8, candidate_size=1000):
        define_img_size(480)
        test_device = "cpu"

        self.__class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.__threshold = threshold
        self.__candidate_size = candidate_size
        self.faceNetModel = create_Mb_Tiny_RFB_fd(len(self.__class_names), is_test=True, device=test_device)
        self.facePretictor = create_Mb_Tiny_RFB_fd_predictor(self.faceNetModel, candidate_size=self.__candidate_size, device=test_device)
        self.faceNetModel.load(self.faceModelPath)

    def face_detector3(self, frame, personCounter):
        time_time = time.time()
        (h, w) = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.facePretictor.predict(image, self.__candidate_size / 2, self.__threshold)
        faceCounter = 0

        faceList = []
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            (startX, startY) = (max(0, int(box[0]) - 10), max(0, int(box[1]) - 10))
            (endX, endY) = (min(w - 1, int(box[2]) + 10), min(h - 1, int(box[3]) + 10))
            faceList.append(frame[startY:endY, startX:endX])

        return faceList

        #     # TODO move
        #     label1, color1 = getMaskStateAnalysis(frame[startY:endY, startX:endX])
        #     label2, color2 = getEmotionsAnalysis2(frame[startY:endY, startX:endX])
        #
        #     label = label1 + " - " + label2
        #     cv2.putText(frame, label, (startX - 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color1, 1)
        #
        #     # cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 1)
        #     # cv2.rectangle(frame, (startX + 2, startY + 2), (endX - 2, endY - 2), color1, 1)
        #     # cv2.rectangle(frame, (startX + 4, startY + 4), (endX - 4, endY - 4), color2, 2)
        #     faceCounter += 1
        # orig_image = cv2.resize(frame, (0, 0), fx=1, fy=1)
        # # cv2.imshow('annotated', orig_image)
        # print("cost time:{}".format(time.time() - time_time))
        #
        # if personCounter > faceCounter:
        #     return personCounter
        # else:
        #     return faceCounter
