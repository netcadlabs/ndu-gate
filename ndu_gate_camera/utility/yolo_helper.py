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

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            y = np.zeros_like(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y

        def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
            # print(boxes.shape)
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

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

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                if min_mode:
                    over = inter / np.minimum(areas[order[0]], areas[order[1:]])
                else:
                    over = inter / (areas[order[0]] + areas[order[1:]] - inter)

                inds = np.where(over <= nms_thresh)[0]
                order = order[inds + 1]

            return np.array(keep)

        nc = prediction[0].shape[1] - 5  # number of classes
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
            box = xywh2xyxy(x[:, :4])

            # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            i, j = (x[:, 5:] > conf_thres).nonzero()
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            x = np.array(np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)).astype(np.float32)

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms_cpu(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]
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
