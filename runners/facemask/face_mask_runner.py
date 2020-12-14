from threading import Thread
import numpy as np
import cv2
import os

import errno
from os import path

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper


class FaceMaskRunner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.__connector_type = connector_type

        self.__dont_use_face_rects = config.get("dont_use_face_rects", False)
        self._last_data = None

        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/model360.onnx"
        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/face_mask.names"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)
        self.class_names = onnx_helper.parse_class_names(class_names_fn)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn, config.get("max_engine_count", 0))

        def generate_anchors(feature_map_sizes_, anchor_sizes_, anchor_ratios_):
            """
            generate anchors.
            :param feature_map_sizes_: list of list, for example: [[40,40], [20,20]]
            :param anchor_sizes_: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
            :param anchor_ratios_: list of list, for example: [[1, 0.5], [1, 0.5]]
            :return:
            """
            anchor_bboxes = []
            for idx, feature_size in enumerate(feature_map_sizes_):
                cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
                cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
                cx_grid, cy_grid = np.meshgrid(cx, cy)
                cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
                cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
                center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

                num_anchors = len(anchor_sizes_[idx]) + len(anchor_ratios_[idx]) - 1
                center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
                anchor_width_heights = []

                # different scales with the first aspect ratio
                for scale in anchor_sizes_[idx]:
                    ratio = anchor_ratios_[idx][0]  # select the first ratio
                    width = scale * np.sqrt(ratio)
                    height = scale / np.sqrt(ratio)
                    anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

                # the first scale, with different aspect ratios (except the first one)
                for ratio in anchor_ratios_[idx][1:]:
                    s1 = anchor_sizes_[idx][0]  # select the first scale
                    width = s1 * np.sqrt(ratio)
                    height = s1 / np.sqrt(ratio)
                    anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

                bbox_coords = center_tiled + np.array(anchor_width_heights)
                bbox_coords_reshape = bbox_coords.reshape((-1, 4))
                anchor_bboxes.append(bbox_coords_reshape)
            anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
            return anchor_bboxes

        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        # feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]] #160 için

        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        self.__anchors_exp = np.expand_dims(anchors, axis=0)

        # # export
        # import torch
        # from torch.autograd import Variable
        # dummy_input = Variable(torch.randn(1, 3, 360, 360))
        # torch.onnx.export(model, dummy_input, "/Users/korhun/Documents/temp/model360.onnx")
        # # https://github.com/ultralytics/yolov5/issues/58
        # # torch==1.5.1 ile çalışıyor!!!!!
        # # export

    def get_name(self):
        return "FaceMaskRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        res = []
        handled = False
        if not self.__dont_use_face_rects and extra_data is not None:
            results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
            if results is not None:
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "face":
                            rect_face = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_face is not None:
                                bbox = rect_face
                                y1 = max(int(bbox[0]), 0)
                                x1 = max(int(bbox[1]), 0)
                                y2 = max(int(bbox[2]), 0)
                                x2 = max(int(bbox[3]), 0)
                                w = x2 - x1
                                h = y2 - y1
                                dw = int(w * 0.25)
                                dh = int(h * 0.25)
                                x1 -= dw
                                x2 += dw
                                y1 -= dh
                                y2 += dh
                                y1 = max(y1, 0)
                                x1 = max(x1, 0)
                                y2 = max(y2, 0)
                                x2 = max(x2, 0)

                                image = frame[y1:y2, x1:x2]
                                self._process(image, res, x1, y1)
                                handled = True
        if not handled:
            self._process(frame, res, 0, 0)

        return res

    def _process(self, image, res, x1, y1):
        def decode_bbox(anchors, raw_outputs):
            variances = [0.1, 0.1, 0.2, 0.2]
            '''
            Decode the actual bbox according to the anchors.
            the anchor value order is:[xmin,ymin, xmax, ymax]
            :param anchors: numpy array with shape [batch, num_anchors, 4]
            :param raw_outputs: numpy array with the same shape with anchors
            :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
            :return:
            '''
            anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
            anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
            anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
            anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
            raw_outputs_rescale = raw_outputs * np.array(variances)
            predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
            predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
            predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
            predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
            predict_xmin = predict_center_x - predict_w / 2
            predict_ymin = predict_center_y - predict_h / 2
            predict_xmax = predict_center_x + predict_w / 2
            predict_ymax = predict_center_y + predict_h / 2
            predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
            return predict_bbox

        def single_class_non_max_suppression(bboxes, confidences, conf_thresh_=0.2, iou_thresh_=0.5, keep_top_k=-1):
            """
            do nms on single class.
            Hint: for the specific class, given the bbox and its confidence,
            1) sort the bbox according to the confidence from top to down, we call this a set
            2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
            3) remove the bbox whose IOU is higher than the iou_thresh from the set,
            4) loop step 2 and 3, util the set is empty.
            :param bboxes: numpy array of 2D, [num_bboxes, 4]
            :param confidences: numpy array of 1D. [num_bboxes]
            :param conf_thresh_:
            :param iou_thresh_:
            :param keep_top_k:
            :return:
            """
            if len(bboxes) == 0:
                return []

            conf_keep_idx = np.where(confidences > conf_thresh_)[0]

            bboxes = bboxes[conf_keep_idx]
            confidences = confidences[conf_keep_idx]

            pick = []
            xmin_ = bboxes[:, 0]
            ymin_ = bboxes[:, 1]
            xmax_ = bboxes[:, 2]
            ymax_ = bboxes[:, 3]

            area = (xmax_ - xmin_ + 1e-3) * (ymax_ - ymin_ + 1e-3)
            idxs = np.argsort(confidences)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                # keep top k
                if keep_top_k != -1:
                    if len(pick) >= keep_top_k:
                        break

                overlap_xmin = np.maximum(xmin_[i], xmin_[idxs[:last]])
                overlap_ymin = np.maximum(ymin_[i], ymin_[idxs[:last]])
                overlap_xmax = np.minimum(xmax_[i], xmax_[idxs[:last]])
                overlap_ymax = np.minimum(ymax_[i], ymax_[idxs[:last]])
                overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
                overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
                overlap_area = overlap_w * overlap_h
                overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

                need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh_)[0]))
                idxs = np.delete(idxs, need_to_be_deleted_idx)

            return conf_keep_idx[pick]

        conf_thresh = 0.5
        # conf_thresh = 0.2
        iou_thresh = 0.4

        target_shape = (360, 360)
        # target_shape = (160, 160)

        height, width, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image_resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA).astype(np.float32)
        image_resized = image_helper.resize_best_quality(image, target_shape).astype(np.float32)
        image_np = image_resized / 255.0
        image_exp = np.expand_dims(image_np, axis=0)

        image_transposed = image_exp.transpose((0, 3, 1, 2))

        y_bboxes_output, y_cls_output = onnx_helper.run(self.sess_tuple, [image_transposed])

        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = decode_bbox(self.__anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                     bbox_max_scores,
                                                     conf_thresh_=conf_thresh,
                                                     iou_thresh_=iou_thresh,
                                                     )

        # # #test
        # item.pop(constants.RESULT_KEY_CLASS_NAME)
        # item.pop(constants.RESULT_KEY_SCORE)
        # item.pop(constants.RESULT_KEY_RECT)

        count_no_mask = 0
        count_mask = 0
        for idx in keep_idxs:
            score = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]

            if class_id == 0:
                count_mask += 1
            else:
                count_no_mask += 1

            xmin = max(0, int(bbox[0] * width)) + x1
            ymin = max(0, int(bbox[1] * height)) + y1
            xmax = min(int(bbox[2] * width), width) + x1
            ymax = min(int(bbox[3] * height), height) + y1

            rect_face = [ymin, xmin, ymax, xmax]
            res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: self.class_names[class_id], constants.RESULT_KEY_SCORE: score})

        data = {constants.RESULT_KEY_DATA: {"mask": count_mask, "no_mask": count_no_mask}}
        if self._last_data != data:
            self._last_data = data
            res.append(data)