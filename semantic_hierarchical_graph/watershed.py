from typing import Dict, List, Tuple
import numpy as np
import cv2
from matplotlib import pyplot as plt
import largestinteriorrectangle as lir
from shapely import Polygon, LineString
from semantic_hierarchical_graph.environment import Environment


def marker_controlled_watershed(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    unknown = cv2.subtract(sure_bg, sure_fg)  # type: ignore

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)  # type: ignore

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    ws = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # show_imgs(img, ws)

    return ws, dist_transform


def largest_rectangle_per_region(ws: np.ndarray, base_size: Tuple[int, int] = (10, 30)) -> Tuple[Dict[int, List], Dict[int, Environment]]:
    ws_tmp = ws.copy()
    largest_rectangles: Dict[int, List] = {}
    region_envs: Dict[int, Environment] = {}
    for i in range(2, ws_tmp.max() + 1):
        first_loop = True
        env = Environment(i, np.array([0, 0, ws.shape[0], ws.shape[1]]))
        region_envs[i] = env
        while True:
            region_bool = np.where(ws_tmp == i, True, False)
            region = region_bool.astype("uint8") * 255
            # TODO: Changed from cv2.RETR_TREE to EXTERNAL because it is faster and hierarchy doesn't matter
            contours, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour_areas = list(map(cv2.contourArea, contours))
            c_max_index = np.argmax(contour_areas)
            c_max = contours[c_max_index]

            if np.max(contour_areas) < 9 * base_size[0] * base_size[1]:
                # print("New rectangle too small", cv2.contourArea(c_max))
                break

            if first_loop:
                first_loop = False

                x, y, w, h = cv2.boundingRect(c_max)
                # interior polygon (= free room) has to be inverted with [::-1] to be clear space in shapely
                walls = Polygon([(x-1, y-1), (x-1, y+h), (x+w, y+h), (x+w, y-1)], [c_max[:, 0, :][::-1]])
                env.add_obstacle(walls)
                obstacle_index = np.where(hierarchy[0, :, 3] == c_max_index)[0]
                [env.add_obstacle(Polygon(contours[index][:, 0, :])) for index in obstacle_index]

            rectangle = lir.lir(region_bool, c_max[:, 0, :])

            if i in largest_rectangles:
                largest_rectangles[i].append(rectangle)
            else:
                largest_rectangles[i] = [rectangle]

            path = path_from_rectangle(rectangle, base_size)
            if not path.is_empty:
                env.add_path(path)

            ws_tmp = cv2.rectangle(ws_tmp, rectangle, (1), -1)

        # env.plot()
        # show_imgs(region)

    # show_imgs(ws_tmp)
    return largest_rectangles, region_envs


def path_from_rectangle(rectangle: np.ndarray, base_size: Tuple[int, int]) -> LineString:
    x, y, w, h = rectangle

    if w < base_size[0] or h < base_size[0]:
        print("corridor too small")
        return LineString()

    if w < 2 * base_size[0] or h < 2 * base_size[0]:
        if w < h:
            point_1 = (x + w//2, y+base_size[0])
            point_2 = (x + w//2, y+h-base_size[0])
        else:
            point_1 = (x + base_size[0], y+h//2)
            point_2 = (x + w-base_size[0], y+h//2)

        # cv2.line(ws, point_1, point_2, (25), 2)
        return LineString([point_1, point_2])
    else:
        point_1 = (x + base_size[0], y + base_size[0])
        point_2 = (x + w - base_size[0], y + base_size[0])
        point_3 = (x + w - base_size[0], y + h - base_size[0])
        point_4 = (x + base_size[0], y + h - base_size[0])
        # rect = (x + base_size[0], y + base_size[0], w - 2 * base_size[0], h - 2 * base_size[0])
        # ws = cv2.rectangle(ws, rect, (25), 2)
        return LineString([point_1, point_2, point_3, point_4, point_1])


def find_bridge_nodes(ws: np.ndarray, dist_transform: np.ndarray) -> Dict[Tuple, List]:
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
    for adj_marker, edge in adjacent_edges.items():
        border = np.zeros(shape=ws.shape, dtype="uint8")
        for pixel in edge:
            border[pixel] = 255
        ret, connected_edges = cv2.connectedComponents(border)

        for i in range(1, connected_edges.max() + 1):
            connected_edge = np.where(connected_edges == i)
            connected_edge = list(zip(connected_edge[0], connected_edge[1]))
            bridge_pixel = max(connected_edge, key=lambda x: dist_transform[x[0], x[1]])

            if adj_marker in bridge_nodes:
                bridge_nodes[adj_marker].append(bridge_pixel[::-1])
            else:
                bridge_nodes[adj_marker] = [bridge_pixel[::-1]]

    return bridge_nodes


def connect_paths(envs: Dict[int, Environment], bridge_nodes: Dict[Tuple, List]):
    for room, env in envs.items():
        for (n1, n2), bridge_points in bridge_nodes.items():
            if room in [n1, n2]:
                for point in bridge_points:
                    env.add_shortest_connection(point)
        # env.plot()


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
        plt.savefig("data/" + name + '.png')
    else:
        plt.show()


def plot_all_envs(envs: Dict[int, Environment]):
    all_envs = Environment("all", np.array([0, 0]))
    for env in envs.values():
        [all_envs.add_obstacle(obstacle) for obstacle in env.scene]
        [all_envs.add_path(path) for path in env.path]

    all_envs.plot()


if __name__ == '__main__':
    img = cv2.imread('data/map_benchmark_ryu.png')

    ws, dist_transform = marker_controlled_watershed(img)
    bridge_nodes = find_bridge_nodes(ws, dist_transform)
    largest_rectangles, region_envs = largest_rectangle_per_region(ws)
    connect_paths(region_envs, bridge_nodes)
    plot_all_envs(region_envs)

    ws2 = draw(ws, bridge_nodes, (22))
    ws3 = draw(ws2, largest_rectangles, (21))
    show_imgs(ws3, name="map_benchmark_ryu_result", save=False)
