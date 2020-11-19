import numpy as np
import cv2
import onnxruntime as rt
import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants, image_helper

class yolov3_tiny_runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.input_size = config.get("input_size", 416)

        self.onnx_fn = config.get("onnx_fn", "yolov3-tiny.onnx")
        if not os.path.isfile(self.onnx_fn):
            self.onnx_fn = os.path.dirname(os.path.abspath(__file__)) + self.onnx_fn.replace("/", os.path.sep)

        self.classes_filename = config.get("classes_filename", "coco.names")
        if not os.path.isfile(self.classes_filename):
            self.classes_filename = os.path.dirname(os.path.abspath(__file__)) + self.classes_filename.replace("/", os.path.sep)

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        self.yolo_sess, self.yolo_input_name, self.yolo_class_names = self._create_session(self.onnx_fn, self.classes_filename)

    def get_name(self):
        return "yolov3-tiny"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return self._predict(self.yolo_sess, self.yolo_input_name, self.input_size, self.yolo_class_names, frame)

    @staticmethod
    def _predict(sess, input_name, input_size, class_names, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_processed, w, h, nw, nh, dw, dh = yolov3_tiny_runner._image_preprocess(np.copy(image), [input_size, input_size])
        # img_processed, w, h, nw, nh, dw, dh = yolov3_tiny_runner._image_preprocess(np.copy(frame), [input_size, input_size])
        image_data = img_processed[np.newaxis, ...].astype(np.float32)
        image_data = np.transpose(image_data, [0, 3, 1, 2])

        # yolov3-tiny için özel kısım
        img_size = np.array([input_size, input_size], dtype=np.float32).reshape(1, 2)
        boxes, scores, indices = sess.run(None, {input_name: image_data, "image_shape": img_size})
        out_boxes, out_scores, out_classes = yolov3_tiny_runner._postprocess_tiny_yolov3(boxes, scores, indices, class_names)

        out_boxes = yolov3_tiny_runner._remove_padding(out_boxes, w, h, nw, nh, dw, dh)

        res = []
        for i in range(len(out_boxes)):
            res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
        return res

    @staticmethod
    def _image_preprocess(image, target_size, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = image_helper.resize_best_quality(image, (nw, nh))

        image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_padded = image_padded / 255.

        if gt_boxes is None:
            return image_padded, w, h, nw, nh, dw, dh

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_padded, gt_boxes

    @staticmethod
    def _postprocess_tiny_yolov3(boxes, scores, indices, class_names):
        objects_identified = indices.shape[0]
        out_boxes, out_scores, out_classes = [], [], []
        if objects_identified > 0:
            for idx_0 in indices:
                for idx_ in idx_0:
                    class_index = idx_[1]
                    out_classes.append(class_names[class_index])
                    out_scores.append(scores[tuple(idx_)])
                    idx_1 = (idx_[0], idx_[2])
                    out_boxes.append(boxes[idx_1])
        return out_boxes, out_scores, out_classes

    def postprocess_tiny_yoloV3(boxes, scores, indices, class_names):
        objects_identified = indices.shape[0]
        out_boxes, out_scores, out_classes = [], [], []
        if objects_identified > 0:
            for idx_0 in indices:
                for idx_ in idx_0:
                    classIndex = idx_[1]
                    out_classes.append(class_names[classIndex])
                    out_scores.append(scores[tuple(idx_)])
                    idx_1 = (idx_[0], idx_[2])
                    out_boxes.append(boxes[idx_1])
        return out_boxes, out_scores, out_classes

    @staticmethod
    def _remove_padding(bboxes, w, h, nw, nh, dw, dh):
        rw = w / nw
        rh = h / nh
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            h1 = bbox[0]
            w1 = bbox[1]
            h2 = bbox[2]
            w2 = bbox[3]

            bbox[0] = int((h1 - dh) * rh)
            bbox[1] = int((w1 - dw) * rw)
            bbox[2] = int((h2 - dh) * rh)
            bbox[3] = int((w2 - dw) * rw)

        return bboxes

    @staticmethod
    def _create_session(onnx_fn, classes_filename):
        class_names = [line.rstrip('\n') for line in open(classes_filename)]

        sess = rt.InferenceSession(onnx_fn)
        input_name = sess.get_inputs()[0].name  # "input_1"
        return sess, input_name, class_names
