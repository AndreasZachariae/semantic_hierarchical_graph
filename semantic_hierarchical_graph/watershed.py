from typing import Dict, List, Tuple
import numpy as np
import cv2
from matplotlib import pyplot as plt
import largestinteriorrectangle as lir


def marker_controlled_watershed(img: np.ndarray):
    # convert to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 0.5 * dist_transform.max()
    # TODO: find a better threshold value
    ret, sure_fg = cv2.threshold(dist_transform, 30, 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # show_imgs(img, markers)

    return markers, dist_transform


def largest_rectangle_per_region(markers: np.ndarray, base_size: Tuple[int, int] = (10, 30)):
    # print(markers.max())
    # region_bool = np.where(markers == 10, True, False)

    original_markers = markers.copy()

    for i in range(2, markers.max() + 1):
        while True:
            region_bool = np.where(markers == i, True, False)
            region = region_bool.astype("uint8") * 255
            contours, _ = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c_max = max(contours, key=cv2.contourArea)

            if cv2.contourArea(c_max) < 9 * base_size[0] * base_size[1]:
                # print("New rectangle too small", cv2.contourArea(c_max))
                break

            contour = c_max[:, 0, :]
            rectangle = lir.lir(region_bool, contour)

            markers = cv2.rectangle(markers, rectangle, (1), -1)
            original_markers = cv2.rectangle(original_markers, rectangle, (21), 2)
            # original_markers = path_from_rectangle(rectangle, original_markers, base_size)

    show_imgs(original_markers)


def path_from_rectangle(rectangle: np.ndarray, markers: np.ndarray, base_size: Tuple[int, int]) -> np.ndarray:
    x, y, w, h = rectangle

    if w < base_size[0] or h < base_size[0]:
        print("corridor too small")
        return markers

    if w < 2 * base_size[0] or h < 2 * base_size[0]:
        if w < h:
            point_1 = (x + w//2, y+base_size[0])
            point_2 = (x + w//2, y+h-base_size[0])
        else:
            point_1 = (x + base_size[0], y+h//2)
            point_2 = (x + w-base_size[0], y+h//2)

        cv2.line(markers, point_1, point_2, (25), 2)
    else:
        rect = (x + base_size[0], y + base_size[0], w - 2 * base_size[0], h - 2 * base_size[0])
        markers = cv2.rectangle(markers, rect, (25), 2)

    return markers


def find_bridge_nodes(markers: np.ndarray, dist_transform: np.ndarray):
    max_row, max_col = markers.shape
    edges = np.where(markers == -1)
    adjacent = np.eye(markers.max()+1)
    # np.zeros(shape=(markers.max()+1, markers.max()+1))
    adjacent_edges: List[List[List]] = [[[] for i in range(markers.max()+1)] for j in range(markers.max()+1)]
    for edge_row, edge_col in zip(edges[0], edges[1]):
        neighbors = set()
        if edge_row-1 >= 0:
            neighbors.add(markers[edge_row-1, edge_col])
        if edge_row+1 < max_row:
            neighbors.add(markers[edge_row+1, edge_col])
        if edge_col-1 >= 0:
            neighbors.add(markers[edge_row, edge_col-1])
        if edge_col+1 < max_col:
            neighbors.add(markers[edge_row, edge_col+1])

        neighbors.discard(1)
        neighbors.discard(-1)

        if len(neighbors) != 2:
            continue

        n1, n2 = list(neighbors)
        if n2 < n1:
            n1, n2 = n2, n1

        adjacent[n1, n2] = 1
        # adjacent[n2, n1] = 1

        if adjacent_edges[n1][n2] == []:
            adjacent_edges[n1][n2] = [(edge_row, edge_col)]
        else:
            adjacent_edges[n1][n2].append((edge_row, edge_col))

        # print("Bridge node found at", (edge_row, edge_col), "with neighbors", neighbors)

    print("Adjacent matrix", adjacent)
    # print("Adjacent edges", adjacent_edges)

    for rows in adjacent_edges:
        for col in rows:
            for edge in col:
                if edge != []:
                    markers[edge[0], edge[1]] = 30

    show_imgs(markers, dist_transform)


def show_imgs(img: np.ndarray, img_2: np.ndarray = None):
    if img_2 is not None:
        plt.subplot(211), plt.imshow(img)
        plt.subplot(212), plt.imshow(img_2)
    else:
        plt.imshow(img)
    plt.show()
    # plt.savefig('data/map_benchmark_ryu_result.png')


if __name__ == '__main__':
    img = cv2.imread('data/map_benchmark_ryu.png')

    markers, dist_transform = marker_controlled_watershed(img)
    find_bridge_nodes(markers, dist_transform)
    # largest_rectangle_per_region(markers)
    # show_imgs(markers)
