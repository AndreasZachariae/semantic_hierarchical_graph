import itertools
from shapely.plotting import plot_polygon, plot_line
import numpy as np
import cv2
from shapely import Polygon, LineString
from semantic_hierarchical_graph.environment import Environment

if __name__ == '__main__':
    # Test combinations iterator
    seq = ["a", "b", "c"]
    for tuple in itertools.combinations(seq, 2):
        print(tuple)
    for tuple in itertools.permutations(seq, 2):
        print(tuple)

    # Test orthogonal line
    env = Environment(0)
    line = LineString([(5, 7), (10, 10)])
    print(env.is_orthogonal_to_grid(line))

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
