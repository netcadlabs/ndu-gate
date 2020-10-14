import errno
import os
import time
from os import path
import cv2

from ndu_gate_camera.detectors.vision.ssd.config.fd_config import define_img_size
from ndu_gate_camera.detectors.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

# from ndu_gate_camera.utility.constants import NDU_GATE_MODEL_FOLDER


from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants


class face_detector_runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.__threshold = config.get("threshold", 0.8)
        self.__candidate_size = config.get("candidate_size", 1000)

        self.label_path = path.dirname(path.abspath(__file__)) + "/data/voc-model-labels.txt"
        self.face_model_path = path.dirname(path.abspath(__file__)) + "/data/version-RFB-640.pth"

        # TODO  koray sil
        # from torch.autograd import Variable
        # import torch.onnx
        # import torchvision
        import torch
        #
        # dummy_input = Variable(torch.randn(1, 3, 256, 256))
        # state_dict = torch.load(self.face_model_path)
        # model = load_state_dict(state_dict)
        # torch.onnx.export(model, dummy_input, "moment-in-time.onnx")
        #
        # from torch.autograd import Variable
        #
        # # Load the trained model from file
        #
        # from torch.testing._internal.data.network2 import Net
        # trained_model = Net()
        # trained_model.load_state_dict(torch.load(self.face_model_path))
        #
        # # Export the trained model to ONNX
        # dummy_input = Variable(torch.randn(1, 1, 28, 28))  # one black and white 28 x 28 picture will be the input to the model
        # torch.onnx.export(trained_model, dummy_input, "face_detection.onnx")








        if not path.isfile(self.label_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.label_path)

        if not path.isfile(self.face_model_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.face_model_path)

        define_img_size(480)
        test_device = "cpu"

        self.__class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.faceNetModel = create_Mb_Tiny_RFB_fd(len(self.__class_names), is_test=True, device=test_device)
        self.facePredictor = create_Mb_Tiny_RFB_fd_predictor(self.faceNetModel, candidate_size=self.__candidate_size, device=test_device)
        self.faceNetModel.load(self.face_model_path)

        # # TODO koray - onnx export denemesi
        # from torch.autograd import Variable
        # dummy_input = Variable(torch.randn(16, 3, 3, 3))  # one black and white 28 x 28 picture will be the input to the model
        # torch.onnx.export(self.faceNetModel, dummy_input, "face_detection.onnx")
        # # export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL,
        # #        input_names=None, output_names=None, aten=False, export_raw_ir=False,
        # #        operator_export_type=None, opset_version=None, _retain_param_name=True,
        # #        do_constant_folding=True, example_outputs=None, strip_doc_string=True,
        # #        dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
        # #        enable_onnx_checker=True, use_external_data_format=False):
        # pass



    def get_name(self):
        return "facedetector"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return self._face_detector3(frame)

    # @staticmethod
    # def _predict(sess, input_name, input_size, class_names, frame):
    #     img_processed, w, h, nw, nh, dw, dh = yolov3_runner._image_preprocess(np.copy(frame), [input_size, input_size])
    #     image_data = img_processed[np.newaxis, ...].astype(np.float32)
    #     image_data = np.transpose(image_data, [0, 3, 1, 2])
    #
    #     # yolov3-tiny için özel kısım
    #     img_size = np.array([input_size, input_size], dtype=np.float32).reshape(1, 2)
    #     boxes, scores, indices = sess.run(None, {input_name: image_data, "image_shape": img_size})
    #     out_boxes, out_scores, out_classes, length = yolov3_runner._postprocess_yolov3(boxes, scores, indices, class_names)
    #
    #     out_boxes = yolov3_runner._remove_padding(out_boxes, w, h, nw, nh, dw, dh)
    #
    #     res = []
    #     for i in range(len(out_boxes)):
    #         res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
    #     return res

    def _face_detector3(self, frame):
        time_time = time.time()
        (h, w) = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.facePredictor.predict(image, self.__candidate_size / 2, self.__threshold)

        res = []
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            (startX, startY) = (max(0, int(box[0]) - 10), max(0, int(box[1]) - 10))
            (endX, endY) = (min(w - 1, int(box[2]) + 10), min(h - 1, int(box[3]) + 10))
            # face_list.append(frame[startY:endY, startX:endX])
            rect = [startY, startX, endY, endX]
            res.append({constants.RESULT_KEY_RECT: rect})

        return res
