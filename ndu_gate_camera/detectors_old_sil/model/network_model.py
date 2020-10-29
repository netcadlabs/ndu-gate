# Already trained model available @
# https://github.com/tensorflow/models/tree/master/research/object_detection
# was used as a part of this code.

import tensorflow as tf
import cv2
import numpy as np

from ndu_gate_camera.detectors_old_sil.model.backbone import set_model


class NetworkModel:
    def __init__(self):
        # detection_graph, self.category_index = backbone.set_model('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        #                                                           'mscoco_label_map.pbtxt')
        # # detection_graph, self.category_index = backbone.set_model(
        # #     'faster_rcnn_resnet50_coco_2018_01_28',
        # #     'mscoco_label_map.pbtxt')
        # detection_graph, self.category_index = set_model("ssd_mobilenet_v1_coco_2018_01_28", "mscoco_label_map.pbtxt")
        detection_graph, self.category_index = set_model("frozen_inference_graph.pb", "mscoco_label_map.pbtxt")
        self.sess = tf.compat.v1.InteractiveSession(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        self.detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        self.detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
        self.num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    def get_category_index(self):
        return self.category_index

    def detect_pedestrians(self, frame):
        # Actual detection.
        # input_frame = cv2.resize(frame, (350, 200))
        input_frame = frame

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_np_expanded},
        )

        classes = np.squeeze(classes).astype(np.int32)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        pedestrian_score_threshold = 0.25
        pedestrian_boxes = []
        total_pedestrians = 0
        imageList = []
        for i in range(int(num[0])):
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]["name"]
                # print(class_name)
                if class_name == "person" and scores[i] > pedestrian_score_threshold:
                    (frame_height, frame_width) = input_frame.shape[:2]
                    image = input_frame.copy()
                    total_pedestrians += 1
                    score_pedestrian = scores[i]
                    pedestrian_boxes.append(boxes[i])
                    # print(np.squeeze(boxes)[i])

                    ymin = int(boxes[i][0] * 0.9 * frame_height)
                    xmin = int(boxes[i][1] * 0.9 * frame_width)
                    ymax = int(boxes[i][2] * 1.1 * frame_height)
                    xmax = int(boxes[i][3] * 1.1 * frame_width)
                    # print(frame_height, frame_width)
                    # print("xmin %s ymin %s xmax %s ymax %s", str(xmin), str(ymin), str(xmax), str(ymax))
                    crop_img = image[ymin:ymax, xmin:xmax]
                    imageList.append(crop_img)

        return pedestrian_boxes, total_pedestrians, imageList
