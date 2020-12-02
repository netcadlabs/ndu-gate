import numpy as np
import cv2
import onnxruntime as rt
import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper, yolo_helper


class Yolov4Runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.input_size = config.get("input_size", 512)

        # self.input_size = 512
        # onnx_fn = "/data/yolov4_-1_3_512_512_dynamic.onnx"
        self.input_size = 608
        onnx_fn = "/data/yolov4_-1_3_608_608_dynamic.onnx"

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = config.get("classes_filename", "coco.names")
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        # self.yolo_sess, self.yolo_input_name, self.yolo_class_names = self._create_session(onnx_fn, classes_filename)
        self.sess_tuple = onnx_helper.create_sess_tuple(onnx_fn)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

    def get_name(self):
        return "yolov4"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
    #     return self._predict(self.yolo_sess, self.yolo_input_name, self.input_size, self.yolo_class_names, frame)
    #
    # @staticmethod
    # def _predict(sess, input_name, input_size, class_names, frame):
    #     h, w, _ = frame.shape
    #
    #     def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    #         x1 = boxes[:, 0]
    #         y1 = boxes[:, 1]
    #         x2 = boxes[:, 2]
    #         y2 = boxes[:, 3]
    #
    #         areas = (x2 - x1) * (y2 - y1)
    #         order = confs.argsort()[::-1]
    #
    #         keep = []
    #         while order.size > 0:
    #             idx_self = order[0]
    #             idx_other = order[1:]
    #
    #             keep.append(idx_self)
    #
    #             xx1 = np.maximum(x1[idx_self], x1[idx_other])
    #             yy1 = np.maximum(y1[idx_self], y1[idx_other])
    #             xx2 = np.minimum(x2[idx_self], x2[idx_other])
    #             yy2 = np.minimum(y2[idx_self], y2[idx_other])
    #
    #             w = np.maximum(0.0, xx2 - xx1)
    #             h = np.maximum(0.0, yy2 - yy1)
    #             inter = w * h
    #
    #             if min_mode:
    #                 over = inter / np.minimum(areas[order[0]], areas[order[1:]])
    #             else:
    #                 over = inter / (areas[order[0]] + areas[order[1:]] - inter)
    #
    #             inds = np.where(over <= nms_thresh)[0]
    #             order = order[inds + 1]
    #
    #         return np.array(keep)
    #
    #     def post_processing(img, conf_thresh, nms_thresh, output):
    #
    #         # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    #         # num_anchors = 9
    #         # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    #         # strides = [8, 16, 32]
    #         # anchor_step = len(anchors) // num_anchors
    #
    #         # [batch, num, 1, 4]
    #         box_array = output[0]
    #         # [batch, num, num_classes]
    #         confs = output[1]
    #
    #         if type(box_array).__name__ != 'ndarray':
    #             box_array = box_array.cpu().detach().numpy()
    #             confs = confs.cpu().detach().numpy()
    #
    #         num_classes = confs.shape[2]
    #
    #         # [batch, num, 4]
    #         box_array = box_array[:, :, 0]
    #
    #         # [batch, num, num_classes] --> [batch, num]
    #         max_conf = np.max(confs, axis=2)
    #         max_id = np.argmax(confs, axis=2)
    #
    #         bboxes_batch = []
    #         for i in range(box_array.shape[0]):
    #
    #             argwhere = max_conf[i] > conf_thresh
    #             l_box_array = box_array[i, argwhere, :]
    #             l_max_conf = max_conf[i, argwhere]
    #             l_max_id = max_id[i, argwhere]
    #
    #             bboxes = []
    #             # nms for each class
    #             for j in range(num_classes):
    #
    #                 cls_argwhere = l_max_id == j
    #                 ll_box_array = l_box_array[cls_argwhere, :]
    #                 ll_max_conf = l_max_conf[cls_argwhere]
    #                 ll_max_id = l_max_id[cls_argwhere]
    #
    #                 keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
    #
    #                 if (keep.size > 0):
    #                     ll_box_array = ll_box_array[keep, :]
    #                     ll_max_conf = ll_max_conf[keep]
    #                     ll_max_id = ll_max_id[keep]
    #
    #                     for k in range(ll_box_array.shape[0]):
    #                         bboxes.append(
    #                             [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
    #                              ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
    #
    #             bboxes_batch.append(bboxes)
    #         return bboxes_batch
    #
    #     IN_IMAGE_H = sess.get_inputs()[0].shape[2]
    #     IN_IMAGE_W = sess.get_inputs()[0].shape[3]
    #
    #     # resized = cv2.resize(frame, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    #     resized = image_helper.resize_best_quality(frame, (IN_IMAGE_W, IN_IMAGE_H))
    #     img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #     img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    #     img_in = np.expand_dims(img_in, axis=0)
    #     img_in /= 255.0
    #
    #     input_name = sess.get_inputs()[0].name
    #
    #     outputs = sess.run(None, {input_name: img_in})
    #
    #     boxes = post_processing(img_in, 0.4, 0.6, outputs)
    #
    #     def process_boxes(boxes, width, height, class_names):
    #         out_boxes1 = []
    #         out_scores1 = []
    #         out_classes1 = []
    #         for box in boxes[0]:
    #             if len(box) >= 7:
    #                 x1 = int(box[0] * width)
    #                 y1 = int(box[1] * height)
    #                 x2 = int(box[2] * width)
    #                 y2 = int(box[3] * height)
    #                 out_boxes1.append([y1, x1, y2, x2])
    #                 out_scores1.append(box[5])
    #                 out_classes1.append(class_names[box[6]])
    #         return out_boxes1, out_scores1, out_classes1
    #
    #     out_boxes, out_scores, out_classes = process_boxes(boxes, w, h, class_names)
    #
    #     res = []
    #     for i in range(len(out_boxes)):
    #         res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i],
    #                     constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
    #     return res
    #
    # @staticmethod
    # def _image_preprocess(image, target_size, gt_boxes=None):
    #     ih, iw = target_size
    #     h, w, _ = image.shape
    #
    #     scale = min(iw / w, ih / h)
    #     nw, nh = int(scale * w), int(scale * h)
    #     image_resized = image_helper.resize_best_quality(image, (nw, nh))
    #
    #     image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    #     dw, dh = (iw - nw) // 2, (ih - nh) // 2
    #     image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
    #     image_padded = image_padded / 255.
    #
    #     if gt_boxes is None:
    #         return image_padded, w, h, nw, nh, dw, dh
    #
    #     else:
    #         gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    #         gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    #         return image_padded, gt_boxes
    #
    # @staticmethod
    # def _create_session(onnx_fn, classes_filename):
    #     class_names = [line.rstrip('\n') for line in open(classes_filename, encoding='utf-8')]
    #
    #     sess = rt.InferenceSession(onnx_fn)
    #     input_name = sess.get_inputs()[0].name  # "input_1"
    #     return sess, input_name, class_names
