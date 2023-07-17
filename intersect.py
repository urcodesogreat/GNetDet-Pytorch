# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
import cv2
from typing import List, Tuple


class Point(object):
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: "Point") -> "Point":
        x = self.x + other.x
        y = self.y + other.y
        return self.__class__(x, y)

    def __sub__(self, other: "Point") -> "Point":
        x = self.x - other.x
        y = self.y - other.y
        return self.__class__(x, y)

    def __eq__(self, other: "Point") -> bool:
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    @property
    def pixel(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y + other.y

    def cross(self, other: "Point") -> float:
        return self.x * other.y - self.y * other.x

    def rend(self, img, color: Tuple[int, int, int] = (255, 0, 0)):
        cv2.circle(img, self.pixel, 3, color, -1)

    def inside_polygon(self, polygon: List["Point"]):
        p_end = Point(10000, self.y)
        ray_line = Line(self, p_end)
        n_sides = len(polygon)
        n_intersects = 0
        for i in range(n_sides):
            p1, p2 = polygon[i], polygon[(i+1) % n_sides]
            side_line = Line(p1, p2)
            if intersect(ray_line, side_line):
                n_intersects += 1
        return n_intersects % 2 != 0


class Line(object):
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def __repr__(self):
        return f"Line({self.p1} - > {self.p2})"

    @property
    def d(self) -> Point:
        x = self.p2.x - self.p1.x
        y = self.p2.y - self.p1.y
        return Point(x, y)

    def rend(self, img, color: Tuple[int, int, int] = (0, 255, 0)):
        cv2.line(img, self.p1.pixel, self.p2.pixel, color, 1)


def intersect(l1: Line, l2: Line) -> bool:
    r = l1.d
    s = l2.d
    q_p = l2.p1 - l1.p1
    qs_p = l2.p2 - l1.p1

    if r.cross(s) == 0:
        if q_p.cross(r) == 0:
            # Colinear
            t0 = q_p.dot(r) / (r.dot(r) + 1e-8)
            t1 = qs_p.dot(r) / (r.dot(r) + 1e-8)
            if 0 <= t0 <= 1 or 0 <= t1 <= 1:
                # Overlapping
                return True
            else:
                # Disjoint
                return False
        else:
            # Parallel
            return False

    else:
        # Segments meat at point inside range
        t = q_p.cross(s) / r.cross(s)
        u = q_p.cross(r) / r.cross(s)
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True

        # Segments meat at point outside range
        return False
