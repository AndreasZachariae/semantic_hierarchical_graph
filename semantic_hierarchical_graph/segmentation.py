from typing import Dict, List, Tuple
import numpy as np
import cv2
from matplotlib import pyplot as plt
from semantic_hierarchical_graph.parameters import Parameter


def marker_controlled_watershed(img: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # convert to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    img_for_ws = cv2.cvtColor(opening.copy(), cv2.COLOR_GRAY2BGR)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # TODO: find a better threshold value
    # currently: 3 * base_size[0] = 30
    # alternative: 3 * safety_distance = 30
    ret, sure_fg = cv2.threshold(dist_transform, params["distance_threshold"], 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)  # type: ignore

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)  # type: ignore

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0

    ws = cv2.watershed(img_for_ws, markers)
    img_for_ws[markers == -1] = [255, 0, 0]

    # Add safety margin as erosion
    size = 1+2*params["safety_distance"]
    print("Add safety distance with erosion and kernel:", (size, size))
    kernel_sd = np.ones((size, size), np.uint8)
    img_with_erosion = cv2.erode(opening.copy(), kernel_sd).astype(np.int32)
    img_with_erosion *= ws

    # show_imgs(img_with_erosion, ws)

    return ws, img_with_erosion, dist_transform


def find_bridge_nodes(ws: np.ndarray, dist_transform: np.ndarray):  # -> Dict[Tuple, List]:
    max_row, max_col = ws.shape
    edges = np.where(ws == -1)
    adjacent = np.eye(ws.max()+1)
    adjacent_edges: Dict[Tuple, List] = {}
    for edge_row, edge_col in zip(edges[0], edges[1]):
        neighbors = set()
        if edge_row-1 >= 0:
            neighbors.add(ws[edge_row-1, edge_col])
        if edge_row+1 < max_row:
            neighbors.add(ws[edge_row+1, edge_col])
        if edge_col-1 >= 0:
            neighbors.add(ws[edge_row, edge_col-1])
        if edge_col+1 < max_col:
            neighbors.add(ws[edge_row, edge_col+1])

        neighbors.discard(1)
        neighbors.discard(-1)

        if len(neighbors) != 2:
            continue

        n1, n2 = list(neighbors)
        if n2 < n1:
            n1, n2 = n2, n1

        adjacent[n1, n2] = 1

        if (n1, n2) in adjacent_edges:
            adjacent_edges[(n1, n2)].append((edge_row, edge_col))
        else:
            adjacent_edges[(n1, n2)] = [(edge_row, edge_col)]

    # print("Adjacent matrix", adjacent)
    bridge_nodes: Dict[Tuple, List] = {}
    bridge_edges: Dict[Tuple, List] = {}
    for adj_marker, edge in adjacent_edges.items():
        border = np.zeros(shape=ws.shape, dtype="uint8")
        for pixel in edge:
            border[pixel] = 255
        ret, connected_edges = cv2.connectedComponents(border)

        for i in range(1, connected_edges.max() + 1):
            connected_edge = np.where(connected_edges == i)
            connected_edge = list(zip(connected_edge[1], connected_edge[0]))
            bridge_pixel = max(connected_edge, key=lambda x: dist_transform[x[1], x[0]])

            if adj_marker in bridge_nodes:
                bridge_nodes[adj_marker].append(bridge_pixel)
                bridge_edges[adj_marker].append(connected_edge)

            else:
                bridge_nodes[adj_marker] = [bridge_pixel]
                bridge_edges[adj_marker] = [connected_edge]

    return bridge_nodes, bridge_edges


def draw(img: np.ndarray, objects_dict: dict, color) -> np.ndarray:
    img_new = img.copy()
    for key, objects_list in objects_dict.items():
        for object in objects_list:
            if len(object) == 2:
                cv2.circle(img_new, object, 4, color, -1)
            if len(object) == 4:
                cv2.rectangle(img_new, object, color, 2)

    return img_new


def show_imgs(img: np.ndarray, img_2: np.ndarray = None, name: str = None, save=False):  # type: ignore
    if img_2 is not None:
        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(img_2)
    else:
        # img = cv2.applyColorMap(img.astype("uint8"), cv2.COLORMAP_JET)
        plt.imshow(img)

    if save:
        # cv2.imwrite("data/" + name + '.png', img)
        plt.savefig("data/" + name + '.png')
    else:
        plt.show()


if __name__ == '__main__':
    # img = cv2.imread('data/map_benchmark_hou2_clean.png')
    img = cv2.imread('data/map_benchmark_ryu.png')
    params = Parameter("config/ryu_params.yaml").params

    ws, ws_erosion, dist_transform = marker_controlled_watershed(img, params)
    bridge_nodes, bridge_edges = find_bridge_nodes(ws, dist_transform)

    ws2 = draw(ws, bridge_nodes, (22))
    show_imgs(ws2, name="map_benchmark_ryu_erosion", save=False)
