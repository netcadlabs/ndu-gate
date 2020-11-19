import numpy as np
import cv2
import onnxruntime as rt
import os

import errno
from os import path


from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.image_helper import image_helper


class face_detector_runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__threshold = config.get("threshold", 0.8)

        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/version-RFB-640.onnx"
        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/voc-model-labels.txt"

        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)

        # self.__onnx_sess, self.__onnx_input_name, self.__onnx_class_names = face_detector_runner._create_session(onnx_fn, class_names_fn)
        def _create_session(onnx_fn, classes_fn):
            sess = rt.InferenceSession(onnx_fn)
            input_name = sess.get_inputs()[0].name
            outputs = sess.get_outputs()
            output_names = []
            for output in outputs:
                output_names.append(output.name)
            class_names = [line.rstrip('\n') for line in open(classes_fn)]
            return sess, input_name, output_names, class_names

        self.__onnx_sess, self.__onnx_input_name, self.__onnx_output_names, self.__onnx_class_names = _create_session(onnx_fn, class_names_fn)

        ## onnx export örneği
        # self.__candidate_size = config.get("candidate_size", 1000)
        # import torch.onnx
        # import torchvision
        # import torch
        # from torch.autograd import Variable
        # model = create_Mb_Tiny_RFB_fd(len(self.__class_names), is_test=False, device=test_device)
        # # self.facePredictor = create_Mb_Tiny_RFB_fd_predictor(self.faceNetModel, candidate_size=self.__candidate_size, device=test_device)
        # model.load(self.face_model_path)
        # dummy_input = Variable(torch.randn(16, 3, 3, 3))
        # torch.onnx.export(model, dummy_input, "/Users/korhun/Documents/temp/version-RFB-640.onnx")
        # pass

    def get_name(self):
        return "facedetector"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        nw = 640
        nh = 480
        h, w = frame.shape[:2]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (nw, nh))
        # image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        image = image_helper.resize_best_quality(image, (nw, nh))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.__onnx_sess.run(self.__onnx_output_names, {self.__onnx_input_name: image})

        out_boxes, out_classes, out_scores = face_detector_runner._predict_faces(w, h, confidences, boxes, self.__threshold, self.__onnx_class_names)

        res = []
        for i in range(len(out_boxes)):
            res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
        return res

    @staticmethod
    def _predict_faces(width, height, confidences, boxes, prob_threshold, class_names, iou_threshold=0.3, top_k=-1):
        # https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/run_video_face_detect_onnx.py
        boxes = boxes[0]
        confidences = confidences[0]
        out_boxes = []
        out_names = []
        out_scores = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = face_detector_runner._hard_nms(box_probs,
                                                       iou_threshold=iou_threshold,
                                                       top_k=top_k,
                                                       )
            for box_prob in box_probs:
                y1 = int(box_prob[0] * width)
                x1 = int(box_prob[1] * height)
                y2 = int(box_prob[2] * width)
                x2 = int(box_prob[3] * height)
                out_boxes.append([x1, y1, x2, y2])
                out_scores.append(box_prob[4])
                out_names.append(class_names[class_index])

        return out_boxes, out_names, out_scores


    @staticmethod
    def _hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
        # import vision.utils.box_utils_numpy as box_utils
        def iou_of(boxes0, boxes1, eps=1e-5):
            def area_of(left_top, right_bottom):
                """
                Compute the areas of rectangles given two corners.
                Args:
                    left_top (N, 2): left top corner.
                    right_bottom (N, 2): right bottom corner.
                Returns:
                    area (N): return the area.
                """
                hw = np.clip(right_bottom - left_top, 0.0, None)
                return hw[..., 0] * hw[..., 1]

            """
            Return intersection-over-union (Jaccard index) of boxes.
            Args:
                boxes0 (N, 4): ground truth boxes.
                boxes1 (N or 1, 4): predicted boxes.
                eps: a small number to avoid 0 as denominator.
            Returns:
                iou (N): IoU values.
            """
            overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
            overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

            overlap_area = area_of(overlap_left_top, overlap_right_bottom)
            area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
            area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
            return overlap_area / (area0 + area1 - overlap_area + eps)

        """
        Perform hard non-maximum-supression to filter out boxes with iou greater
        than threshold
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
            picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]
