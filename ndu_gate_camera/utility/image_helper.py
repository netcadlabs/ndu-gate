import math

import cv2
import base64


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

    @staticmethod
    def rescale_frame(frame, percent):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def frame2base64(frame, scale=40):
        scaled_frame = image_helper.rescale_frame(frame, scale)
        res, frame = cv2.imencode('.png', scaled_frame)
        base64_data = base64.b64encode(frame)
        return base64_data.decode('utf-8')
