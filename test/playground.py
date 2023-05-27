import itertools
from matplotlib import pyplot as plt
from shapely.plotting import plot_polygon, plot_line
from shapely.ops import split
import numpy as np
import cv2
from shapely import Point, Polygon, LineString, snap, unary_union
from semantic_hierarchical_graph.environment import Environment


def cut(line, point):
    # Cuts a line in two at a point
    distance = line.project(point)
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


if __name__ == '__main__':
    # Test combinations iterator
    seq = ["a", "b", "c"]
    print("Combinations:")
    for tuple in itertools.combinations(seq, 2):
        print(tuple)
    print("Permutations:")
    for tuple in itertools.permutations(seq, 2):
        print(tuple)

    # Test orthogonal line
    env = Environment(0)
    line = LineString([(5, 7), (10, 10)])
    print(env._is_orthogonal_to_grid(line))

    # Test bounding boy creation from opencv vs shapely
    rectangle = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0]]).astype(np.uint8)

    empty = np.zeros((5, 6)).astype(np.uint8)

    rect = cv2.boundingRect(rectangle)
    print(rect)
    x, y, w, h = rect
    # cv2.rectangle(empty, (x, y), (x+w-1, y+h-1), (2), 1)
    ws_tmp = cv2.rectangle(empty, (x+1, y+1), (x+w-2, y+h-2), (2), -1)

    print(empty)
    point_1 = (x, y)
    point_2 = (x + w - 1, y)
    point_3 = (x + w - 1, y + h - 1)
    point_4 = (x, y + h - 1)
    # ws = cv2.rectangle(ws, rectangle, (25), 2)
    poly = Polygon([point_1, point_2, point_3, point_4, point_1])
    print(poly)
    print(poly.area+w+h-1)
    print(w*h)

    # Test shapely snap
    line = LineString([(170, 157), (204, 160)])
    point = Point(179.6, 157.8)

    # p2 = snap(point, line, 1)
    p2 = line.interpolate(line.project(point))
    print(p2 == point)
    results = split(line, p2)
    print(results.wkt)

    print(line.contains(p2))
    print(line.distance(p2))
    print(line.within(p2))
    print(line.touches(p2))

    print(line.project(point))
    result = cut(line, point)
    print(result)

    # Test union giving non empty GeometryCollection
    polygon = Polygon([(0, 0), (1, 1), (1, 0)])
    union = unary_union([line, polygon])
    print(union)

    fig, ax = plt.subplots()
    plot_line(line)
    plot_line(polygon)
    plot_line(point)
    plot_line(p2)
    plt.show()
