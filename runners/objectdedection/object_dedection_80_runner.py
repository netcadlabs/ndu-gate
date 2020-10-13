import numpy as np
import cv2
import onnxruntime as rt
import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log


class ObjectDedection80Runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.model_type = config.get("model_type", 0)
        self.input_size = config.get("input_size", 416)

        self.onnx_fn = config.get("onnx_fn", "yolov3.onnx")
        if not os.path.isfile(self.onnx_fn):
            self.onnx_fn = os.path.dirname(os.path.abspath(__file__)) + self.onnx_fn.replace("/", os.path.sep)

        self.classes_filename = config.get("classes_filename", "coco.names")
        if not os.path.isfile(self.classes_filename):
            self.classes_filename = os.path.dirname(os.path.abspath(__file__)) + self.classes_filename.replace("/", os.path.sep)

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        self.yolo_sess, self.yolo_input_name, self.yolo_class_names = self.create_session()

    def get_name(self):
        return "ObjectDedection80Runner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)
        log.debug("ObjectDedection80Runner içindeyim")

        result = self.predict(self.yolo_sess, self.yolo_input_name, self.input_size, self.model_type, self.yolo_class_names, frame)

        return result

    def predict(self, sess, input_name, input_size, model_type, class_names, frame):
        img_processed, w, h, nw, nh, dw, dh = self.image_preprocess(np.copy(frame), [input_size, input_size])
        image_data = img_processed[np.newaxis, ...].astype(np.float32)
        image_data = np.transpose(image_data, [0, 3, 1, 2])

        if model_type == 0:  # yolov3.onnx
            img_size = np.array([input_size, input_size], dtype=np.float32).reshape(1, 2)
            boxes, scores, indices = sess.run(None, {input_name: image_data, "image_shape": img_size})
            out_boxes, out_scores, out_classes, length = self.postprocess_yoloV3(boxes, scores, indices, class_names)
        elif model_type == 1:  # yolov3-tiny.onnx  tiny-yolov3-11.onnx
            img_size = np.array([input_size, input_size], dtype=np.float32).reshape(1, 2)
            boxes, scores, indices = sess.run(None, {input_name: image_data, "image_shape": img_size})
            out_boxes, out_scores, out_classes, length = self.postprocess_tiny_yoloV3(boxes, scores, indices, class_names)

        out_boxes = self.remove_padding(out_boxes, w, h, nw, nh, dw, dh)

        rect = "rect"
        score = "score"
        class_name = "class_name"
        res = []
        for i in range(len(out_boxes)):
            res.append({rect:out_boxes[i], score:out_scores[i], class_name:class_names[i]})
        return res

    def image_preprocess(self, image, target_size, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

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

    def postprocess_yoloV3(self, boxes, scores, indices, class_names):
        objects_identified = indices.shape[0]
        len = 0
        out_boxes, out_scores, out_classes = [], [], []
        if objects_identified > 0:
            for idx_ in indices:
                class_index = idx_[1]
                # if class_index==0 or class_index==67: #person - cell phone
                out_classes.append(class_names[class_index])
                out_scores.append(scores[tuple(idx_)])
                idx_1 = (idx_[0], idx_[2])
                out_boxes.append(boxes[idx_1])
                len = len + 1
        return out_boxes, out_scores, out_classes, len

    def postprocess_tiny_yoloV3(self, boxes, scores, indices, class_names):
        objects_identified = indices.shape[0]
        len = 0
        out_boxes, out_scores, out_classes = [], [], []
        if objects_identified > 0:
            for idx_0 in indices:
                for idx_ in idx_0:
                    class_index = idx_[1]
                    if class_index == 0:
                        out_classes.append(class_names[class_index])
                        out_scores.append(scores[tuple(idx_)])
                        idx_1 = (idx_[0], idx_[2])
                        out_boxes.append(boxes[idx_1])
                        len = len + 1
        return out_boxes, out_scores, out_classes, len

    def remove_padding(self, bboxes, w, h, nw, nh, dw, dh):
        rw = w / nw
        rh = h / nh
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            h1 = bbox[0]
            w1 = bbox[1]
            h2 = bbox[2]
            w2 = bbox[3]

            bbox[0] = (h1 - dh) * rh
            bbox[1] = (w1 - dw) * rw
            bbox[2] = (h2 - dh) * rh
            bbox[3] = (w2 - dw) * rw

        return bboxes

    def create_session(self):
        class_names = [line.rstrip('\n') for line in open(self.classes_filename)]

        sess = rt.InferenceSession(self.onnx_fn)
        input_name = sess.get_inputs()[0].name  # "input_1"
        return sess, input_name, class_names
