import math


def is_inside_polygon(polygon, point):
    # ref:https://stackoverflow.com/a/2922778/1266873
    # int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
    # {
    #   int i, j, c = 0;
    #   for (i = 0, j = nvert-1; i < nvert; j = i++) {
    #     if ( ((verty[i]>testy) != (verty[j]>testy)) &&
    #      (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
    #        c = !c;
    #   }
    #   return c;
    # }
    def pnpoly(nvert, vertx, verty, testx, testy):
        i = 0
        j = nvert - 1
        c = False
        while True:
            j = i
            i += 1
            if i >= nvert:
                break
            if (verty[i] > testy) != (verty[j] > testy) and testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (
                    verty[j] - verty[i]) + vertx[i]:
                c = not c
        return c

    vertx = []
    verty = []
    for p in polygon:
        vertx.append(p[0])
        verty.append(p[1])
    if polygon[-1] is not polygon[0]:
        p = polygon[0]
        vertx.append(p[0])
        verty.append(p[1])
    return pnpoly(len(vertx), vertx, verty, point[0], point[1])


def is_inside_rect(rect, point):
    e = 0.001
    x = point[0]
    if x < rect[1] - e:
        return False
    elif x > rect[3] + e:
        return False
    y = point[1]
    if y < rect[0] - e:
        return False
    elif y > rect[2] + e:
        return False
    return True


def rects_intersect(rect1, rect2):
    class Rectangle:
        def intersects(self, other):
            a, b = self, other
            x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
            y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
            x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
            y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
            return x1 < x2 and y1 < y2

        def _set(self, x1, y1, x2, y2):
            if x1 > x2 or y1 > y2:
                raise ValueError("Coordinates are invalid")
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

        def __init__(self, bbox):
            self._set(bbox[0], bbox[1], bbox[2], bbox[3])

    return Rectangle(rect1).intersects(Rectangle(rect2))


def rects_overlap(rect1, rect2, min_ratio):
    w1 = rect1[2] - rect1[0]
    w2 = rect2[2] - rect2[0]
    w_min = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    if (w_min / w1) >= min_ratio or (w_min / w2) >= min_ratio:
        h1 = rect1[3] - rect1[1]
        h2 = rect2[3] - rect2[1]
        h_min = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
        return (h_min / h1) >= min_ratio or (h_min / h2) >= min_ratio
    return False


def add_padding_rect(rect, padding):
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    dw = (x2 - x1) * padding
    dh = (y2 - y1) * padding
    return [x1 - dw, y1 - dh, x2 + dw, y2 + dh]


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
