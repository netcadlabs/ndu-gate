import math


class geometry_helper:
    @staticmethod
    def is_inside(polygon, point):
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
                if (verty[i]>testy) != (verty[j]>testy) and testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]:
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
        return pnpoly(len(vertx), vertx, verty, point[0], point[1] )

    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

