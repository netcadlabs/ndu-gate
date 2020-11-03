import math
import cv2
import base64
import numpy as np


class image_helper:
    # Boyut değişikliğine en uygun interpolation yöntemi ile resize eder.
    @staticmethod
    def resize_best_quality(image, size):
        size0 = max(image.shape[0], image.shape[1])
        size1 = max(size[0], size[1])
        if size0 > size1:
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        else:
            return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    # İmaj boyutlarından birisi max_dim'den daha büyükse küçültür, değilse aynen döner.
    @staticmethod
    def resize_if_larger(image, max_dim, interpolation=None):
        h, w = image.shape[:2]
        if w > h:
            if w > max_dim:
                return image_helper.resize(image, width=max_dim)
            else:
                return image
        else:
            if h > max_dim:
                return image_helper.resize(image, height=max_dim)
            else:
                return image

    # 'width' veya 'height yoksa en-boy oranını koruyarak resize eder. İkisi de varsa normal resize eder.
    # 'interpolation' yoksa en uygununu seçer.
    @staticmethod
    def resize(image, width=None, height=None, interpolation=None):
        if width is None and height is None:
            return image

        h, w = image.shape[:2]
        dim = None
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)

        if interpolation is None:
            return image_helper.resize_best_quality(image, dim)
        else:
            return cv2.resize(image, dim, interpolation=interpolation)

    # total_pixel_count sonucun width * height değeridir.
    # int yuvarlaması yüzünden sonuç w*h değer, tam total_pixel_count olmayabilir.
    @staticmethod
    def resize_total_pixel_count(image, total_pixel_count):
        h, w = image.shape[:2]
        ratio = w / float(h)
        w1 = math.sqrt(total_pixel_count * ratio)
        h1 = w1 * h / float(w)
        return image_helper.resize_best_quality(image, (int(w1), int(h1)))

    # OpenCV mat nesnesini base64 string yapar
    @staticmethod
    def to_base64(image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer)

    # base64 string'i OpenCV mat nesnesi yapar
    @staticmethod
    def from_base64(base64_text):
        original = base64.b64decode(base64_text)
        as_np = np.frombuffer(original, dtype=np.uint8)
        return cv2.imdecode(as_np, flags=1)

    @staticmethod
    def fill_polyline_transparent(image, pnts, color, opacity, thickness=-1):
        blk = np.zeros(image.shape, np.uint8)
        cv2.drawContours(blk, pnts, -1, color, -1)
        if thickness >= 0:
            cv2.polylines(image, pnts, True, color=color, thickness=thickness)
        res = cv2.addWeighted(image, 1.0, blk, 0.1, 0)
        cv2.copyTo(res, None, image)

    @staticmethod
    def select_areas(frame, window_name, color=(0, 0, 255), opacity=0.3, thickness=4, max_count=None, next_area_key="n", finish_key="s"):
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
                    image_helper.fill_polyline_transparent(image, [pts], color=color, opacity=opacity, thickness=thickness)

                if not new_area:
                    if len(area) > 0:
                        pts = np.array(area, np.int32)
                        image_helper.fill_polyline_transparent(image, [pts], color=color, opacity=opacity, thickness=thickness)
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

            if len(area) > 2:
                areas.append(area)
            return areas
        finally:
            cv2.destroyWindow(window_name)

    @staticmethod
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

        #
        #
        #
        # lines = []
        # line = []
        #
        # def get_mouse_points(event, x, y, _flags, _param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         if len(line) < 2:
        #             cv2.circle(frame, (x, y), 10, (0, 255, 255), 10)
        #             line.append((x, y))
        #
        # cv2.namedWindow(window_name)
        # cv2.moveWindow(window_name, 40, 30)
        # cv2.setMouseCallback(window_name, get_mouse_points)
        #
        # while True:
        #     for ln in lines:
        #         pts = np.array(ln, np.int32)
        #         cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)
        #
        #     cv2.imshow(window_name, frame)
        #     k = cv2.waitKey(1)
        #     if k & 0xFF == ord("s"):
        #         cv2.destroyWindow(window_name)
        #         break
        #     if len(line) == 2:
        #         lines.append(line)
        #         line = []
        #
        # return lines

    @staticmethod
    def put_text(img, text_, center, color=None, font_scale=0.5, thickness=1, back_color=None):
        if back_color is None:
            back_color = [0, 0, 0]
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

    @staticmethod
    def rescale_frame(frame, percent):
        width = int(frame.shape[1] * percent / 100.0)
        height = int(frame.shape[0] * percent / 100.0)
        dim = (width, height)
        return image_helper.resize_best_quality(frame, dim)

    @staticmethod
    def frame2base64(frame, scale=40):
        scaled_frame = image_helper.rescale_frame(frame, scale)
        res, frame = cv2.imencode('.png', scaled_frame)
        base64_data = base64.b64encode(frame)
        return base64_data.decode('utf-8')
