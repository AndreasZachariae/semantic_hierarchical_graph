from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from shapely.plotting import plot_polygon, plot_line
from shapely.ops import unary_union, polygonize_full

import numpy as np
import cv2
import largestinteriorrectangle as lir
from shapely import Polygon, LineString, MultiPolygon
from semantic_hierarchical_graph.environment import Environment
import semantic_hierarchical_graph.segmentation as segmentation


def largest_rectangle_per_segment(ws_erosion: np.ndarray, safety_distance: int) -> Tuple[Dict[int, List], Dict[int, Environment]]:
    ws_tmp = ws_erosion.copy()
    largest_rectangles: Dict[int, List] = {}
    segment_envs: Dict[int, Environment] = {}
    for i in range(2, ws_tmp.max() + 1):
        first_loop = True
        env = Environment(i, np.array([0, 0, ws_tmp.shape[0], ws_tmp.shape[1]]))
        segment_envs[i] = env
        while True:
            segment_bool = np.where(ws_tmp == i, True, False)
            segment = segment_bool.astype("uint8") * 255
            # TODO: Changed from cv2.RETR_TREE to EXTERNAL because it is faster and hierarchy doesn't matter
            contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour_areas = list(map(cv2.contourArea, contours))
            c_max_index = np.argmax(contour_areas)
            c_max = contours[c_max_index]

            # TODO: find better threshold specification
            # currently 9 * 10 * 30 = 2700
            if np.max(contour_areas) < 4 * (2*safety_distance)**2:
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

            rectangle = lir.lir(segment_bool, c_max[:, 0, :])

            if i in largest_rectangles:
                largest_rectangles[i].append(rectangle)
            else:
                largest_rectangles[i] = [rectangle]

            path = path_from_rectangle(rectangle, safety_distance)
            if not path.is_empty:
                env.add_path(path)

            x, y, w, h = rectangle
            # Shrink rectangle by x pixel to make touching possible
            shrinkage = 1
            ws_tmp = cv2.rectangle(ws_tmp, (x+shrinkage, y+shrinkage), (x+w-1-shrinkage, y+h-1-shrinkage), (1), -1)

        # env.plot()
        # segmentation.show_imgs(segment)

    # show_imgs(ws_tmp)
    return largest_rectangles, segment_envs


def path_from_rectangle(rectangle: np.ndarray, safety_distance: int) -> LineString:
    x, y, w, h = rectangle

    if w < 2*safety_distance or h < 2*safety_distance:
        print("corridor too small")
        return LineString()

    # Offset of -1 pixel needed because opencv retuns w, h instead of coordinates
    # But shapely takes 1 pixel width edge as infinte thin line,
    # thus rectangles with edge 1 pixel different are not touching (not able to merge).
    # So rectangles are shrinked by 1 pixel in ws_tmp to compensate and make touching possible.

    w -= 1
    h -= 1
    if w < 3 * safety_distance or h < 3 * safety_distance:
        if w < h:
            point_1 = (x + w//2, y)
            point_2 = (x + w//2, y+h)
        else:
            point_1 = (x, y+h//2)
            point_2 = (x + w, y+h//2)

        # cv2.line(ws, point_1, point_2, (25), 2)
        return LineString([point_1, point_2])
    else:
        point_1 = (x, y)
        point_2 = (x + w, y)
        point_3 = (x + w, y + h)
        point_4 = (x, y + h)
        # ws = cv2.rectangle(ws, rectangle, (25), 2)
        return LineString([point_1, point_2, point_3, point_4, point_1])


def connect_paths(envs: Dict[int, Environment], bridge_nodes: Dict[Tuple, List], bridge_edges: Dict[Tuple, List]):
    for room, env in envs.items():
        if not env.path:
            continue
        print("Connecting paths in room", room)
        result, dangles, cuts, invalids = polygonize_full(env.path)
        union_polygon: Polygon = unary_union(result)
        if isinstance(union_polygon, MultiPolygon):
            env.path = [poly.boundary for poly in union_polygon.geoms]
        elif isinstance(union_polygon, Polygon):
            env.path = [union_polygon.boundary]
        else:
            raise Exception("unknown shape returned from polygon union")
        for cut in cuts.geoms:
            env.add_path(cut)
        if len(dangles.geoms) > 0 or len(invalids.geoms):
            raise Exception("unhandled dangles or invalids are not added to env.path")

        connections = env.find_all_shortest_connections()
        for connection in connections:
            env.add_path(connection)

        for (n1, n2), bridge_points in bridge_nodes.items():
            if room in [n1, n2]:
                # env.clear_bridge_nodes(bridge_points)
                env.clear_bridge_edges(bridge_edges[(n1, n2)])
                for point in bridge_points:
                    connection = env.find_shortest_connection(point)
                    if connection is not None:
                        env.add_path(connection)
                    else:
                        print("No connection found for bridge node", point)
        # env.plot()


def plot_all_envs(envs: Dict[int, Environment]):
    all_envs = Environment("all", np.array([0, 0]))
    for env in envs.values():
        [all_envs.add_obstacle(obstacle) for obstacle in env.scene]
        [all_envs.add_path(path) for path in env.path]

    all_envs.plot()


if __name__ == '__main__':
    # img = cv2.imread('data/map_benchmark_hou2_clean.png')
    img = cv2.imread('data/map_benchmark_ryu.png')
    safety_distance = segmentation.get_safety_distance(base_size=(10, 20), safety_margin=5)

    ws, ws_erosion, dist_transform = segmentation.marker_controlled_watershed(img, safety_distance)
    bridge_nodes, bridge_edges = segmentation.find_bridge_nodes(ws, dist_transform)
    largest_rectangles, segment_envs = largest_rectangle_per_segment(ws_erosion, safety_distance)
    connect_paths(segment_envs, bridge_nodes, bridge_edges)
    plot_all_envs(segment_envs)

    ws2 = segmentation.draw(ws, bridge_nodes, (22))
    ws3 = segmentation.draw(ws2, largest_rectangles, (21))
    segmentation.show_imgs(ws3, name="map_benchmark_ryu_result", save=False)