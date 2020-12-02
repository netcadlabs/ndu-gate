import numpy as np
import cv2
import time

from ndu_gate_camera.utility import constants, onnx_helper, image_helper


def predict_v5(sess_tuple, input_size, class_names, frame):
    # https://github.com/ultralytics/yolov5
    # torch ve torchvision bağımlılığını kaldırmak için birçok değişiklik yapılmıştır, kod üstteki linktekinden değişiktir.

    def image_preprocess(image1, target_size):
        ih, iw = target_size
        h1, w1, _ = image1.shape

        scale = min(iw / w1, ih / h1)
        nw, nh = int(scale * w1), int(scale * h1)
        image_resized = image_helper.resize_best_quality(image1, (nw, nh))

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

    res = []
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_processed, w, h = image_preprocess(np.copy(image), [input_size, input_size])
    image_data = img_processed[np.newaxis, ...].astype(np.float32)
    image_data = np.transpose(image_data, [0, 3, 1, 2])

    inputs = [image_data]
    pred = onnx_helper.run(sess_tuple, inputs)[0]

    batch_detections = np.array(pred)
    batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    detections = batch_detections[0]
    if detections is not None:
        labels = detections[..., -1]
        boxs = detections[..., :4]
        confs = detections[..., 4]
        boxs[:, :] = scale_coords((input_size, input_size), boxs[:, :], (h, w)).round()
        for i, box in enumerate(boxs):
            x1, y1, x2, y2 = box
            class_name = class_names[int(labels[i])]
            score = confs[i]
            res.append({constants.RESULT_KEY_RECT: [y1, x1, y2, x2],
                        constants.RESULT_KEY_SCORE: score, constants.RESULT_KEY_CLASS_NAME: class_name})

    # for i in range(len(out_boxes)):
    #    res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
    return res


def predict_v4(sess_tuple, input_size, class_names, frame):
    h, w, _ = frame.shape

    def nms_cpu(boxes_, confs, nms_thresh=0.5, min_mode=False):
        x1 = boxes_[:, 0]
        y1 = boxes_[:, 1]
        x2 = boxes_[:, 2]
        y2 = boxes_[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

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

    def post_processing(conf_thresh, nms_thresh, output):

        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # num_anchors = 9
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]
        # anchor_step = len(anchors) // num_anchors

        # [batch, num, 1, 4]
        box_array = output[0]
        # [batch, num, num_classes]
        confs = output[1]

        if type(box_array).__name__ != 'ndarray':
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        bboxes_batch = []
        for i_ in range(box_array.shape[0]):
            argwhere = max_conf[i_] > conf_thresh
            l_box_array = box_array[i_, argwhere, :]
            l_max_conf = max_conf[i_, argwhere]
            l_max_id = max_id[i_, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):

                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                             ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

            bboxes_batch.append(bboxes)
        return bboxes_batch

    # IN_IMAGE_H = sess.get_inputs()[0].shape[2]
    # IN_IMAGE_W = sess.get_inputs()[0].shape[3]
    IN_IMAGE_H = IN_IMAGE_W = input_size

    # resized = cv2.resize(frame, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    resized = image_helper.resize_best_quality(frame, (IN_IMAGE_W, IN_IMAGE_H))
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # input_name = sess.get_inputs()[0].name
    # outputs = sess.run(None, {input_name: img_in})
    inputs = [img_in]
    outputs = onnx_helper.run(sess_tuple, inputs)

    boxes = post_processing(0.4, 0.6, outputs)

    def process_boxes(boxes_, width, height, class_names_):
        out_boxes1 = []
        out_scores1 = []
        out_classes1 = []
        for box in boxes_[0]:
            if len(box) >= 7:
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
                out_boxes1.append([y1, x1, y2, x2])
                out_scores1.append(box[5])
                out_classes1.append(class_names_[box[6]])
        return out_boxes1, out_scores1, out_classes1

    out_boxes, out_scores, out_classes = process_boxes(boxes, w, h, class_names)

    res = []
    for i in range(len(out_boxes)):
        res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i],
                    constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
    return res
