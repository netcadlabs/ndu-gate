import os
import time

import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper, yolo_helper, image_helper, constants

# os.environ['Path'] += \
#     "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\ngraph\lib;" \
#     "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\inference_engine\\external\\tbb\\bin;" \
#     "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\inference_engine\\external\\hddl\\bin;" \
#     "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\inference_engine\\bin\\intel64\\Release;" \
#     "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\inference_engine\\bin\\intel64\\Debug;" \
#     "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\model_optimizer;"


# from openvino.inference_engine import IECore
# import ngraph as ng
import cv2
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork


class Yolov5sRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        onnx_fn = "/data/yolov5s.onnx"
        classes_filename = "/data/coco.names"
        self.input_size = 640

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)

    def get_name(self):
        return "yolov5s"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v5(self.sess_tuple, self.input_size, self.class_names, frame)


#
# class Yolov5sRunner_openvino(NDUCameraRunner):
#     def __init__(self, config, connector_type):
#         super().__init__()
#         bin_fn = "/data/yolov5s.bin"
#         xml_fn = "/data/yolov5s.xml"
#         onnx_fn = "/data/yolov5s.onnx"
#         classes_filename = "/data/coco.names"
#         self.input_size = 640
#
#         if not os.path.isfile(onnx_fn):
#             onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)
#         if not os.path.isfile(bin_fn):
#             bin_fn = os.path.dirname(os.path.abspath(__file__)) + bin_fn.replace("/", os.path.sep)
#         if not os.path.isfile(xml_fn):
#             xml_fn = os.path.dirname(os.path.abspath(__file__)) + xml_fn.replace("/", os.path.sep)
#         if not os.path.isfile(classes_filename):
#             classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
#         self.class_names = onnx_helper.parse_class_names(classes_filename)
#
#         ie = IECore()
#
#         # ---1. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format ---
#         # model = xml_fn
#         # self.net = net = ie.read_network(model=model)
#         self.net = net = ie.read_network(onnx_fn)
#         func = ng.function_from_cnn(net)
#         ops = func.get_ordered_ops()
#         # -----------------------------------------------------------------------------------------------------
#
#         # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
#         # versions = ie.get_versions(args.device)
#         # print("{}{}".format(" " * 8, args.device))
#         # print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
#         #                                                       versions[args.device].minor))
#         # print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))
#         #
#         # if args.cpu_extension and "CPU" in args.device:
#         #     ie.add_extension(args.cpu_extension, "CPU")
#         #     log.info("CPU extension loaded: {}".format(args.cpu_extension))
#         # ie.add_extension("CPU", "CPU")
#         # -----------------------------------------------------------------------------------------------------
#
#         # --------------------------- 3. Read and preprocess input --------------------------------------------
#
#         # print("inputs number: " + str(len(net.input_info.keys())))
#         #
#         # for input_key in net.input_info:
#         #     print("input shape: " + str(net.input_info[input_key].input_data.shape))
#         #     print("input key: " + input_key)
#         #     if len(net.input_info[input_key].input_data.layout) == 4:
#         #         n, c, h, w = net.input_info[input_key].input_data.shape
#         #
#         # images = np.ndarray(shape=(n, c, h, w))
#         # images_hw = []
#         # for i in range(n):
#         #     image = cv2.imread(args.input[i])
#         #     ih, iw = image.shape[:-1]
#         #     images_hw.append((ih, iw))
#         #     log.info("File was added: ")
#         #     log.info("        {}".format(args.input[i]))
#         #     if (ih, iw) != (h, w):
#         #         log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
#         #         image = cv2.resize(image, (w, h))
#         #     image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#         #     images[i] = image
#
#         # -----------------------------------------------------------------------------------------------------
#
#         # --------------------------- 4. Configure input & output ---------------------------------------------
#         # --------------------------- Prepare input blobs -----------------------------------------------------
#         # log.info("Preparing input blobs")
#         assert (len(net.input_info.keys()) == 1 or len(
#             net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
#         self.out_blob = next(iter(net.outputs))
#         self.input_name, input_info_name = "", ""
#
#         for input_key in net.input_info:
#             if len(net.input_info[input_key].layout) == 4:
#                 self.input_name = input_key
#                 print("Batch size is {}".format(net.batch_size))
#                 net.input_info[input_key].precision = 'U8'
#             elif len(net.input_info[input_key].layout) == 2:
#                 input_info_name = input_key
#                 net.input_info[input_key].precision = 'FP32'
#                 if net.input_info[input_key].input_data.shape[1] != 3 and net.input_info[input_key].input_data.shape[1] != 6 or \
#                         net.input_info[input_key].input_data.shape[0] != 1:
#                     print('Invalid input info. Should be 3 or 6 values length.')
#
#         # data = {}
#         # data[input_name] = images
#         #
#         # if input_info_name != "":
#         #     infos = np.ndarray(shape=(n, c), dtype=float)
#         #     for i in range(n):
#         #         infos[i, 0] = h
#         #         infos[i, 1] = w
#         #         infos[i, 2] = 1.0
#         #     data[input_info_name] = infos
#
#         # --------------------------- Prepare output blobs ----------------------------------------------------
#         # log.info('Preparing output blobs')
#
#         output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
#         output_ops = {op.friendly_name: op for op in ops \
#                       if op.friendly_name in net.outputs and op.get_type_name() == "DetectionOutput"}
#         if len(output_ops) != 0:
#             output_name, output_info = output_ops.popitem()
#
#         # if output_name == "":
#         #     log.error("Can't find a DetectionOutput layer in the topology")
#
#         output_dims = output_info.shape
#         if len(output_dims) != 4:
#             print("Incorrect output dimensions for SSD model")
#         max_proposal_count, object_size = output_dims[2], output_dims[3]
#
#         if object_size != 7:
#             print("Output item should have 7 as a last dimension")
#
#         output_info.precision = "FP32"
#         # -----------------------------------------------------------------------------------------------------
#
#         # --------------------------- Performing inference ----------------------------------------------------
#         # log.info("Loading model to the device")
#         self.exec_net = ie.load_network(network=net, device_name="CPU")
#         # log.info("Creating infer request and starting inference")
#
#         # -----------------------------------------------------------------------------------------------------
#
#         self.input_info_name = input_info_name
#
#     def get_name(self):
#         return "yolov5s"
#
#     def get_settings(self):
#         settings = {}
#         return settings
#
#     def process_frame(self, frame, extra_data=None):
#         super().process_frame(frame)
#
#         net = self.net
#         for input_key in net.input_info:
#             print("input shape: " + str(net.input_info[input_key].input_data.shape))
#             print("input key: " + input_key)
#             if len(net.input_info[input_key].input_data.layout) == 4:
#                 n, c, h, w = net.input_info[input_key].input_data.shape
#
#         images = np.ndarray(shape=(n, c, h, w))
#         images_hw = []
#         for i in range(n):
#             # image = cv2.imread(args.input[i])
#             image = frame
#             ih, iw = image.shape[:-1]
#             images_hw.append((ih, iw))
#             if (ih, iw) != (h, w):
#                 image = cv2.resize(image, (w, h))
#             image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#             images[i] = image
#
#         data = {}
#         data[self.input_name] = images
#
#         if self.input_info_name != "":
#             infos = np.ndarray(shape=(n, c), dtype=float)
#             for i in range(n):
#                 infos[i, 0] = h
#                 infos[i, 1] = w
#                 infos[i, 2] = 1.0
#             data[self.input_info_name] = infos
#
#         res = self.exec_net.infer(inputs=data)
#         # -----------------------------------------------------------------------------------------------------
#
#         # --------------------------- Read and postprocess output ---------------------------------------------
#         res = res[self.out_blob]
#         boxes, classes = {}, {}
#         data = res[0][0]
#         for number, proposal in enumerate(data):
#             if proposal[2] > 0:
#                 imid = np.int(proposal[0])
#                 ih, iw = images_hw[imid]
#                 label = np.int(proposal[1])
#                 confidence = proposal[2]
#                 xmin = np.int(iw * proposal[3])
#                 ymin = np.int(ih * proposal[4])
#                 xmax = np.int(iw * proposal[5])
#                 ymax = np.int(ih * proposal[6])
#                 print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
#                       .format(number, label, confidence, xmin, ymin, xmax, ymax, imid), end="")
#                 if proposal[2] > 0.5:
#                     print(" WILL BE PRINTED!")
#                     if not imid in boxes.keys():
#                         boxes[imid] = []
#                     boxes[imid].append([xmin, ymin, xmax, ymax])
#                     if not imid in classes.keys():
#                         classes[imid] = []
#                     classes[imid].append(label)
#                 else:
#                     print()
#
#         for imid in classes:
#             tmp_image = frame.copy()
#             for box in boxes[imid]:
#                 cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
#             cv2.imwrite("out.bmp", tmp_image)
#             cv2.imshow("aaaa", frame)
#             cv2.waitKey(0)


class Yolov5sRunner1(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        bin_fn = "/data/yolov5s.bin"
        xml_fn = "/data/yolov5s.xml"
        onnx_fn = "/data/yolov5s.onnx"
        classes_filename = "/data/coco.names"
        self.input_size = 640

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)
        if not os.path.isfile(bin_fn):
            bin_fn = os.path.dirname(os.path.abspath(__file__)) + bin_fn.replace("/", os.path.sep)
        if not os.path.isfile(xml_fn):
            xml_fn = os.path.dirname(os.path.abspath(__file__)) + xml_fn.replace("/", os.path.sep)
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

        ie = IECore()
        net = IENetwork(model=xml_fn, weights=bin_fn)
        # net = ie.read_network(onnx_fn)

        self.exec_net = ie.load_network(network=net, device_name="CPU")
        self.input_blob = next(iter(net.inputs))
        # self.outputs = net.outputs.keys()

    def get_name(self):
        return "yolov5s"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):

        def image_preprocess(image1, target_size):
            ih, iw = target_size
            h1, w1, _ = image1.shape

            scale = min(iw / w1, ih / h1)
            nw, nh = int(scale * w1), int(scale * h1)
            if nh != h1 or nw != w1:
                image_resized = image_helper.resize_best_quality(image1, (nw, nh))
            else:
                image_resized = image1

            image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2
            image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
            image_padded = image_padded / 255.

            return image_padded, w1, h1

        def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
            """Performs Non-Maximum Suppression (NMS) on inference results

            Returns:
                 detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
            """

            def xywh2xyxy(x_):
                # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
                y = np.zeros_like(x_)
                y[:, 0] = x_[:, 0] - x_[:, 2] / 2  # top left x
                y[:, 1] = x_[:, 1] - x_[:, 3] / 2  # top left y
                y[:, 2] = x_[:, 0] + x_[:, 2] / 2  # bottom right x
                y[:, 3] = x_[:, 1] + x_[:, 3] / 2  # bottom right y
                return y

            def nms_cpu(boxes_, confs_, nms_thresh=0.5, min_mode=False):
                # print(boxes.shape)
                x1_ = boxes_[:, 0]
                y1_ = boxes_[:, 1]
                x2_ = boxes_[:, 2]
                y2_ = boxes_[:, 3]

                areas = (x2_ - x1_) * (y2_ - y1_)
                order = confs_.argsort()[::-1]

                keep = []
                while order.size > 0:
                    idx_self = order[0]
                    idx_other = order[1:]

                    keep.append(idx_self)

                    xx1 = np.maximum(x1_[idx_self], x1_[idx_other])
                    yy1 = np.maximum(y1_[idx_self], y1_[idx_other])
                    xx2 = np.minimum(x2_[idx_self], x2_[idx_other])
                    yy2 = np.minimum(y2_[idx_self], y2_[idx_other])

                    w_ = np.maximum(0.0, xx2 - xx1)
                    h_ = np.maximum(0.0, yy2 - yy1)
                    inter = w_ * h_

                    if min_mode:
                        over = inter / np.minimum(areas[order[0]], areas[order[1:]])
                    else:
                        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

                    inds = np.where(over <= nms_thresh)[0]
                    order = order[inds + 1]

                return np.array(keep)

            # nc = prediction[0].shape[1] - 5  # number of classes
            xc = prediction[..., 4] > conf_thres  # candidates

            # Settings
            min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
            max_det = 300  # maximum number of detections per image
            time_limit = 10.0  # seconds to quit after

            t = time.time()
            output = [None] * prediction.shape[0]
            for xi, x in enumerate(prediction):  # image index, image inference
                # Apply constraints
                # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
                x = x[xc[xi]]  # confidence

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box_ = xywh2xyxy(x[:, :4])

                # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                i_, j = (x[:, 5:] > conf_thres).nonzero()
                # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
                x = np.array(np.concatenate((box_[i_], x[i_, j + 5, None], j[:, None]), 1)).astype(np.float32)

                # If none remain process next image
                n = x.shape[0]  # number of boxes
                if not n:
                    continue

                # Sort by confidence
                # x = x[x[:, 4].argsort(descending=True)]

                # Batched NMS
                c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
                i_ = nms_cpu(boxes, scores, iou_thres)
                if i_.shape[0] > max_det:  # limit detections
                    i_ = i_[:max_det]

                output[xi] = x[i_]
                if (time.time() - t) > time_limit:
                    break  # time limit exceeded

            return output

        def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

            def clip_coords(boxes, img_shape):
                boxes[:, 0].clip(0, img_shape[1])  # x1
                boxes[:, 1].clip(0, img_shape[0])  # y1
                boxes[:, 2].clip(0, img_shape[1])  # x2
                boxes[:, 3].clip(0, img_shape[0])  # y2

            # Rescale coords (xyxy) from img1_shape to img0_shape
            if ratio_pad is None:  # calculate from img0_shape
                gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
                pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
            else:
                gain = ratio_pad[0][0]
                pad = ratio_pad[1]

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain
            clip_coords(coords, img0_shape)
            return coords

        super().process_frame(frame)

        res = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_processed, w, h = image_preprocess(np.copy(image), [self.input_size, self.input_size])
        image_data = img_processed[np.newaxis, ...].astype(np.float32)
        image_data = np.transpose(image_data, [0, 3, 1, 2])

        inputs = [image_data]

        pred0 = self.exec_net.infer({self.input_blob: inputs})

        # pred = pred0[self.outputs]
        pred = pred0["output"]
        # pred = onnx_helper.run(sess_tuple, inputs)[0]

        batch_detections = np.array(pred)
        batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
        detections = batch_detections[0]
        if detections is not None:
            labels = detections[..., -1]
            boxs = detections[..., :4]
            confs = detections[..., 4]
            boxs[:, :] = scale_coords((self.input_size, self.input_size), boxs[:, :], (h, w)).round()
            for i, box in enumerate(boxs):
                x1, y1, x2, y2 = box
                class_name = self.class_names[int(labels[i])]
                score = confs[i]
                res.append({constants.RESULT_KEY_RECT: [y1, x1, y2, x2],
                            constants.RESULT_KEY_SCORE: score,
                            constants.RESULT_KEY_CLASS_NAME: class_name})

        # for i in range(len(out_boxes)):
        #    res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
        return res


class Yolov5sRunner2(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        onnx_fn = "/data/yolov5s.onnx"
        classes_filename = "/data/coco.names"
        self.input_size = 640

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)
        # self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)
        self.sess, self.input_names, self.output_names, onnx_fn= onnx_helper.get_sess_tuple(onnx_fn)
        self.input_name = self.input_names[0]
        self.input_item = {self.input_name: None}

    def get_name(self):
        return "yolov5s"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):

        def image_preprocess(image1, target_size):
            ih, iw = target_size
            h1, w1, _ = image1.shape

            scale = min(iw / w1, ih / h1)
            nw, nh = int(scale * w1), int(scale * h1)
            if nh != h1 or nw != w1:
                image_resized = image_helper.resize_best_quality(image1, (nw, nh))
            else:
                image_resized = image1

            image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2
            image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
            image_padded = image_padded / 255.

            return image_padded, w1, h1

        def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
            """Performs Non-Maximum Suppression (NMS) on inference results

            Returns:
                 detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
            """

            def xywh2xyxy(x_):
                # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
                y = np.zeros_like(x_)
                y[:, 0] = x_[:, 0] - x_[:, 2] / 2  # top left x
                y[:, 1] = x_[:, 1] - x_[:, 3] / 2  # top left y
                y[:, 2] = x_[:, 0] + x_[:, 2] / 2  # bottom right x
                y[:, 3] = x_[:, 1] + x_[:, 3] / 2  # bottom right y
                return y

            def nms_cpu(boxes_, confs_, nms_thresh=0.5, min_mode=False):
                # print(boxes.shape)
                x1_ = boxes_[:, 0]
                y1_ = boxes_[:, 1]
                x2_ = boxes_[:, 2]
                y2_ = boxes_[:, 3]

                areas = (x2_ - x1_) * (y2_ - y1_)
                order = confs_.argsort()[::-1]

                keep = []
                while order.size > 0:
                    idx_self = order[0]
                    idx_other = order[1:]

                    keep.append(idx_self)

                    xx1 = np.maximum(x1_[idx_self], x1_[idx_other])
                    yy1 = np.maximum(y1_[idx_self], y1_[idx_other])
                    xx2 = np.minimum(x2_[idx_self], x2_[idx_other])
                    yy2 = np.minimum(y2_[idx_self], y2_[idx_other])

                    w_ = np.maximum(0.0, xx2 - xx1)
                    h_ = np.maximum(0.0, yy2 - yy1)
                    inter = w_ * h_

                    if min_mode:
                        over = inter / np.minimum(areas[order[0]], areas[order[1:]])
                    else:
                        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

                    inds = np.where(over <= nms_thresh)[0]
                    order = order[inds + 1]

                return np.array(keep)

            # nc = prediction[0].shape[1] - 5  # number of classes
            xc = prediction[..., 4] > conf_thres  # candidates

            # Settings
            min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
            max_det = 300  # maximum number of detections per image
            time_limit = 10.0  # seconds to quit after

            t = time.time()
            output = [None] * prediction.shape[0]
            for xi, x in enumerate(prediction):  # image index, image inference
                # Apply constraints
                # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
                x = x[xc[xi]]  # confidence

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box_ = xywh2xyxy(x[:, :4])

                # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                i_, j = (x[:, 5:] > conf_thres).nonzero()
                # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
                x = np.array(np.concatenate((box_[i_], x[i_, j + 5, None], j[:, None]), 1)).astype(np.float32)

                # If none remain process next image
                n = x.shape[0]  # number of boxes
                if not n:
                    continue

                # Sort by confidence
                # x = x[x[:, 4].argsort(descending=True)]

                # Batched NMS
                c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
                i_ = nms_cpu(boxes, scores, iou_thres)
                if i_.shape[0] > max_det:  # limit detections
                    i_ = i_[:max_det]

                output[xi] = x[i_]
                if (time.time() - t) > time_limit:
                    break  # time limit exceeded

            return output

        def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

            def clip_coords(boxes, img_shape):
                boxes[:, 0].clip(0, img_shape[1])  # x1
                boxes[:, 1].clip(0, img_shape[0])  # y1
                boxes[:, 2].clip(0, img_shape[1])  # x2
                boxes[:, 3].clip(0, img_shape[0])  # y2

            # Rescale coords (xyxy) from img1_shape to img0_shape
            if ratio_pad is None:  # calculate from img0_shape
                gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
                pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
            else:
                gain = ratio_pad[0][0]
                pad = ratio_pad[1]

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain
            clip_coords(coords, img0_shape)
            return coords

        super().process_frame(frame)

        res = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_processed, w, h = image_preprocess(np.copy(image), [self.input_size, self.input_size])
        image_data = img_processed[np.newaxis, ...].astype(np.float32)
        image_data = np.transpose(image_data, [0, 3, 1, 2])

        # inputs = [image_data]

        # pred = onnx_helper.run(self.sess_tuple, inputs)[0]
        # input_item = {self.input_names[0]: inputs[0]}
        # input_item = {self.input_name: inputs[0]}
        # input_item = {self.input_name: image_data}
        # pred = self.sess.run(self.output_names, input_item)
        # pred = self.sess.run(self.output_names, {self.input_name: image_data})[0]
        self.input_item[self.input_name] = image_data
        pred = self.sess.run(self.output_names, self.input_item)[0]

        batch_detections = np.array(pred)
        batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
        detections = batch_detections[0]
        if detections is not None:
            labels = detections[..., -1]
            boxs = detections[..., :4]
            confs = detections[..., 4]
            boxs[:, :] = scale_coords((self.input_size, self.input_size), boxs[:, :], (h, w)).round()
            for i, box in enumerate(boxs):
                x1, y1, x2, y2 = box
                class_name = self.class_names[int(labels[i])]
                score = confs[i]
                res.append({constants.RESULT_KEY_RECT: [y1, x1, y2, x2],
                            constants.RESULT_KEY_SCORE: score,
                            constants.RESULT_KEY_CLASS_NAME: class_name})

        # for i in range(len(out_boxes)):
        #    res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
        return res
