import math
import cv2
import base64
import numpy as np

from ndu_gate_camera.utility import geometry_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


def image_h_w(image):
    # h, w = image.shape[:2]
    return image.shape[:2]


# Boyut değişikliğine en uygun interpolation yöntemi ile resize eder.
def resize_best_quality(image, size):
    if image.shape[0] == size[0] and image.shape[1] == size[1]:
        return image.copy()
    size0 = max(image.shape[0], image.shape[1])
    size1 = max(size[0], size[1])
    if size0 > size1:
        # if size0 > 2 * size1:
        #     image = cv2.pyrDown(image)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


# İmaj boyutlarından birisi max_dim'den daha büyükse küçültür, değilse aynen döner.
def resize_if_larger(image, max_dim, interpolation=None):
    h, w = image.shape[:2]
    if w > h:
        if w > max_dim:
            return resize(image, width=max_dim)
        else:
            return image
    else:
        if h > max_dim:
            return resize(image, height=max_dim)
        else:
            return image


# İmaj boyutlarından "büyük olan" min_dim'den daha küçükse resize edilir, değilse aynen döner.
def resize_if_smaller(image, min_dim, interpolation=None):
    h, w = image.shape[:2]
    if w > h:
        if w < min_dim:
            return resize(image, width=min_dim)
        else:
            return image
    else:
        if h < min_dim:
            return resize(image, height=min_dim)
        else:
            return image


# 'width' veya 'height yoksa en-boy oranını koruyarak resize eder. İkisi de varsa normal resize eder.
# 'interpolation' yoksa en uygununu seçer.
def resize(image, width=None, height=None, interpolation=None):
    if width is None and height is None:
        return image

    h, w = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    if interpolation is None:
        return resize_best_quality(image, dim)
    else:
        return cv2.resize(image, dim, interpolation=interpolation)


# total_pixel_count sonucun width * height değeridir.
# int yuvarlama yüzünden sonuç w*h değer, tam total_pixel_count olmayabilir.
def resize_total_pixel_count(image, total_pixel_count):
    h, w = image.shape[:2]
    ratio = w / float(h)
    w1 = math.sqrt(total_pixel_count * ratio)
    h1 = w1 * h / float(w)
    return resize_best_quality(image, (int(w1), int(h1)))


# OpenCV mat nesnesini base64 string yapar
def to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer)


# base64 string'i OpenCV mat nesnesi yapar
def from_base64(base64_text):
    original = base64.b64decode(base64_text)
    as_np = np.frombuffer(original, dtype=np.uint8)
    return cv2.imdecode(as_np, flags=1)


def fill_polyline_transparent(image, pnts, color, opacity, thickness=-1):
    blk = np.zeros(image.shape, np.uint8)
    cv2.drawContours(blk, pnts, -1, color, -1)
    if thickness >= 0:
        cv2.polylines(image, pnts, True, color=color, thickness=thickness)
    res = cv2.addWeighted(image, 1.0, blk, 0.1, 0)
    cv2.copyTo(res, None, image)


def select_areas(frame, window_name, color=(0, 0, 255), opacity=0.3, thickness=4, max_count=None, next_area_key="n", finish_key="s", return_tuples=True, max_point_count=None):
    try:
        areas = []
        area = []

        def get_mouse_points(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                area.append((x, y))

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 40, 30)
        cv2.setMouseCallback(window_name, get_mouse_points)

        new_area = False
        while True:
            image = frame.copy()
            for area1 in areas:
                pts = np.array(area1, np.int32)
                fill_polyline_transparent(image, [pts], color=color, opacity=opacity, thickness=thickness)

            if not new_area:
                if len(area) > 0:
                    pts = np.array(area, np.int32)
                    fill_polyline_transparent(image, [pts], color=color, opacity=opacity,
                                              thickness=thickness)
                    for pnt in area:
                        cv2.circle(image, pnt, thickness * 2, color, thickness)
            else:
                if len(area) > 2:
                    areas.append(area)
                if max_count is not None and len(areas) == max_count:
                    return areas
                else:
                    area = []
                    new_area = False

            cv2.imshow(window_name, image)
            k = cv2.waitKey(1)
            if k & 0xFF == ord(finish_key):
                break
            elif k & 0xFF == ord(next_area_key):
                new_area = True
            elif max_point_count is not None and len(area) == max_point_count:
                new_area = True

        if len(area) > 2:
            areas.append(area)
        if not return_tuples:
            for i in range(len(areas)):
                areas[i] = np.array(areas[i], np.int32)
        return areas
    finally:
        cv2.destroyWindow(window_name)


def select_lines(frame, window_name, color=(0, 255, 255), thickness=4, max_count=None, finish_key="s"):
    try:
        lines = []
        line = []

        def get_mouse_points(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                line.append((x, y))

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 40, 30)
        cv2.setMouseCallback(window_name, get_mouse_points)

        while True:
            image = frame.copy()
            for line1 in lines:
                pts = np.array(line1, np.int32)
                cv2.polylines(image, [pts], False, color=color, thickness=thickness)
            for pnt in line:
                cv2.circle(image, pnt, thickness * 2, color, thickness)
            if len(line) == 2:
                lines.append(line)
                if max_count is not None and len(lines) == max_count:
                    return lines
                else:
                    line = []

            cv2.imshow(window_name, image)
            k = cv2.waitKey(1)
            if k & 0xFF == ord(finish_key):
                break

        return lines
    finally:
        cv2.destroyWindow(window_name)


def select_points(frame, window_name, color=(0, 255, 255), radius=8, thickness=4, max_count=None, finish_key="s"):
    try:
        pnts = []

        def get_mouse_points(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pnts.append((x, y))

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 40, 30)
        cv2.setMouseCallback(window_name, get_mouse_points)

        while True:
            image = frame.copy()
            for pnt in pnts:
                cv2.circle(image, pnt, radius, color, thickness)
            cv2.imshow(window_name, image)
            k = cv2.waitKey(1)
            if k & 0xFF == ord(finish_key):
                break
            if max_count is not None and max_count <= len(pnts):
                break
        return pnts
    finally:
        cv2.destroyWindow(window_name)


def put_text(img, text_, center, color=None, font_scale=0.5, thickness=1, back_color=None, replace_tur_chars=True):
    if replace_tur_chars:
        text_ = NDUUtility.debug_replace_tur_chars(text_)
    if back_color is None:
        back_color = [1, 1, 1]
    if color is None:
        color = [255, 255, 255]
    y = center[1]
    # font = cv2.FONT_HERSHEY_COMPLEX
    font = cv2.FONT_HERSHEY_DUPLEX
    coor = (int(center[0] + 5), int(y))
    cv2.putText(img=img, text=text_, org=coor,
                fontFace=font, fontScale=font_scale, color=back_color, lineType=cv2.LINE_AA,
                thickness=thickness + 2)
    cv2.putText(img=img, text=text_, org=coor,
                fontFace=font, fontScale=font_scale, color=color,
                lineType=cv2.LINE_AA, thickness=thickness)


def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100.0)
    height = int(frame.shape[0] * percent / 100.0)
    dim = (width, height)
    return resize_best_quality(frame, dim)


def frame2base64(frame, scale=40):
    scaled_frame = rescale_frame(frame, scale)
    res, frame = cv2.imencode('.png', scaled_frame)
    base64_data = base64.b64encode(frame)
    return base64_data.decode('utf-8')


def normalize_coordinates(image, coords):
    h, w = image.shape[:2]
    pnts = []
    for x, y in coords:
        pnts.append((x / w, y / h))
    return pnts


def denormalize_coordinates(image, coords):
    h, w = image.shape[:2]
    pnts = []
    for x, y in coords:
        pnts.append((int(x * w), int(y * h)))
    return pnts


def normalize_lines(image, lines):
    lines1 = []
    for line in lines:
        lines1.append(normalize_coordinates(image, line))
    return lines1


def denormalize_lines(image, lines):
    lines1 = []
    for line in lines:
        lines1.append(denormalize_coordinates(image, line))
    return lines1


def convert_lines_list2tuple(lines):
    res = []
    for line in lines:
        line1 = []
        for c in line:
            line1.append(tuple(c))
        res.append(line1)
    return res


def convert_lines_tuple2list(lines):
    res = []
    for line in lines:
        line1 = []
        for c in line:
            line1.append(list(c))
        res.append(line1)
    return res


def crop(frame, rect):
    y1 = max(int(rect[0]), 0)
    x1 = max(int(rect[1]), 0)
    y2 = max(int(rect[2]), 0)
    x2 = max(int(rect[3]), 0)
    return frame[y1:y2, x1:x2]


def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = 0 - value
        v[v < lim] = 0
        v[v >= lim] -= abs(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_mask(shape, areas):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    contours = []
    for area in areas:
        contours.append(np.array(area, np.int32))
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask


def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def draw_rect(image, rect, color=None):
    if color is None:
        color = [255, 255, 255]
    c = np.array(rect[:4], dtype=np.int32)
    c1, c2 = geometry_helper.get_rect_pnts(c)
    cv2.rectangle(image, c1, c2, color=[1, 1, 1], thickness=3)
    cv2.rectangle(image, c1, c2, color=color, thickness=2)


def equalize_hist_bgr(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    # return img


def adaptive_equalize_hist_bgr(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img[:, :, 0] = clahe.apply(img[:, :, 0])
    # img[:, :, 1] = clahe.apply(img[:, :, 1])
    # img[:, :, 2] = clahe.apply(img[:, :, 2])
    # return img
