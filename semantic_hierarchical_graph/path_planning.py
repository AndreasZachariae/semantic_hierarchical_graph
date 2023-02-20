import itertools
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import cv2
from shapely.ops import unary_union, polygonize_full
from shapely import Polygon, LineString, MultiPolygon, GeometryCollection, Point
import largestinteriorrectangle as lir
from semantic_hierarchical_graph.environment import Environment
from semantic_hierarchical_graph.parameters import Parameter
import semantic_hierarchical_graph.segmentation as segmentation
from semantic_hierarchical_graph.types import Position


def _create_rooms(ws_erosion: np.ndarray, params: Dict[str, Any]) -> Tuple[Dict[int, Environment], Dict[int, List]]:
    ws_tmp = ws_erosion.copy()
    segment_envs: Dict[int, Environment] = {}
    largest_rectangles: Dict[int, List] = {}
    for i in range(2, ws_tmp.max() + 1):
        env: Environment = Environment(i)
        segment_envs[i] = env
        largest_rectangles[i], _ = calc_largest_rectangles(ws_tmp, env, params)
    return segment_envs, largest_rectangles


def calc_largest_rectangles(ws_erosion: np.ndarray, env: Environment, params: Dict[str, Any]) -> Tuple[List, Position]:
    largest_rectangles: List = []
    centroid = Position(0, 0, 0)
    first_loop = True
    while True:
        segment = np.where(ws_erosion == env.room_id, 255, 0).astype("uint8")
        # TODO: Changed from cv2.RETR_TREE to EXTERNAL because it is faster and hierarchy doesn't matter
        contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_areas = list(map(cv2.contourArea, contours))
        c_max_index = np.argmax(contour_areas)
        c_max = contours[c_max_index]

        if first_loop:
            first_loop = False
            M = cv2.moments(segment)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = Position(cX, cY, 0)

            x, y, w, h = cv2.boundingRect(c_max)
            # interior polygon (= free room) has to be inverted with [::-1] to be clear space in shapely
            walls = Polygon([(x-1, y-1), (x-1, y+h), (x+w, y+h), (x+w, y-1)], [c_max[:, 0, :][::-1]])
            env.add_obstacle(walls)
            obstacle_index = np.where(hierarchy[0, :, 3] == c_max_index)[0]
            [env.add_obstacle(Polygon(contours[index][:, 0, :])) for index in obstacle_index]

        elif np.max(contour_areas) < params["min_contour_area"]:
            # TODO: find better threshold specification
            # currently 9 * 10 * 30 = 2700
            # print("New contour too small", cv2.contourArea(c_max))
            break

        rectangle = lir.lir(segment.astype("bool"), c_max[:, 0, :])

        largest_rectangles.append(rectangle)

        path = _path_from_rectangle(rectangle, params)
        if not path.is_empty:
            env.add_path(path)

        x, y, w, h = rectangle
        # Shrink rectangle by x pixel to make touching possible
        shrinkage = 1
        ws_erosion = cv2.rectangle(ws_erosion, (x+shrinkage, y+shrinkage),
                                   (x+w-1-shrinkage, y+h-1-shrinkage), (1), -1)

    # env.plot()
    # segmentation.show_imgs(segment, ws_erosion)
    return largest_rectangles, centroid


def _path_from_rectangle(rectangle: np.ndarray, params: dict) -> LineString:
    x, y, w, h = rectangle

    if w < params["min_corridor_width"] or h < params["min_corridor_width"]:
        # print("corridor too small")
        return LineString()

    # Offset of -1 pixel needed because opencv retuns w, h instead of coordinates
    # But shapely takes 1 pixel width edge as infinte thin line,
    # thus rectangles with edge 1 pixel different are not touching (not able to merge).
    # So rectangles are shrinked by 1 pixel in ws_tmp to compensate and make touching possible.

    w -= 1
    h -= 1
    if w < params["max_rectangle_to_line_width"] or h < params["max_rectangle_to_line_width"]:
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


def _create_paths(envs: Dict[int, Environment], bridge_nodes: Dict[Tuple, List], bridge_edges: Dict[Tuple, List], params: dict):
    for room, env in envs.items():
        room_bridge_nodes = {adj_rooms: points for adj_rooms, points in bridge_nodes.items() if room in adj_rooms}
        room_bridge_edges = {adj_rooms: points for adj_rooms, points in bridge_edges.items() if room in adj_rooms}
        connect_paths(env, room_bridge_nodes, room_bridge_edges, params)


def connect_paths(env: Environment, bridge_nodes: Dict[Tuple, List], bridge_edges: Dict[Tuple, List], params: dict) -> Set:
    # Get all bridge points of this room to others
    bridge_points = [point for points in bridge_nodes.values() for point in points]
    bridge_points_not_connected = set()
    if not env.scene:
        print("No scene and paths in room", env.room_id)
        bridge_points_not_connected.update(bridge_points)
        return bridge_points_not_connected

    # Remove all bridge edges from walls
    [env.clear_bridge_edges(edge_points) for edge_points in bridge_edges.values()]

    # Connect bridge points between each other
    if not env.path:
        print("No path, connect only bridge points in room", env.room_id)
        for p1, p2 in itertools.combinations(bridge_points, 2):
            connection = env.get_valid_connection(Point(p1[0], p1[1]), Point(p2[0], p2[1]))
            if connection is None:
                bridge_points_not_connected.add(p1)
                bridge_points_not_connected.add(p2)
                continue
            else:
                env.add_path(connection)
                print("Connection between bridge points added")
        if not env.path:
            print("No path in room", env.room_id)
            bridge_points_not_connected.update(bridge_points)
            return bridge_points_not_connected

    # Merge rectangles and connect bridge points to path
    else:
        print("Connecting paths in room", env.room_id)
        result, dangles, cuts, invalids = polygonize_full(env.path)
        union_polygon: Polygon = unary_union(result)
        if isinstance(union_polygon, MultiPolygon):
            env.path = [poly.boundary for poly in union_polygon.geoms]
        elif isinstance(union_polygon, Polygon):
            env.path = [union_polygon.boundary]
        elif isinstance(union_polygon, GeometryCollection):
            # union polygon is empty. No polygon in original room path
            pass
        else:
            raise Exception("unknown shape returned from polygon union")
        if len(dangles.geoms) > 0 or len(invalids.geoms):
            raise Exception("unhandled dangles or invalids are not added to env.path")
        for cut in cuts.geoms:
            env.add_path(cut)

        connections = env.find_all_shortest_connections(mode="without_polygon", polygon=union_polygon)
        # connections = env.find_all_vertex_connections(mode="without_polygon", polygon=union_polygon)
        for connection in connections:
            env.add_path(connection)

        for point in bridge_points:
            connections, _ = connect_point_to_path(point, env, params)
            if len(connections) > 0:
                for connection in connections:
                    env.add_path(connection)
            else:
                bridge_points_not_connected.add(point)
    # print(len(env.path), "paths in room", env.room_id)
    # env.plot()
    return bridge_points_not_connected


def connect_point_to_path(point: Tuple[float, float], env: Environment, params: dict) -> Tuple[List[LineString], Any]:
    """ List of connections is always in direction from point to path.
        Every connection is a two point line without collision.
    """
    connection, closest_path = env.find_shortest_connection(
        point, params["max_attempts_to_connect_bridge_point_straight"])
    if connection is None:
        # TODO: find connection with auxillary points
        # TODO: find connection with A* algorithm
        print("No connection found point", point)
        return [], None

    return [connection], closest_path


def _plot_all_envs(envs: Dict[int, Environment]):
    all_envs = Environment(-1)
    for env in envs.values():
        [all_envs.add_obstacle(obstacle) for obstacle in env.scene]
        [all_envs.add_path(path) for path in env.path]

    all_envs.plot()


def _draw_all_paths(img: np.ndarray, envs: Dict[int, Environment],  color) -> np.ndarray:
    img_new = img.copy()
    all_envs = Environment(-1)
    for env in envs.values():
        [cv2.polylines(img_new, [line.coords._coords.astype("int32")], False,  color, 2) for line in env.path]

    return img_new


if __name__ == '__main__':

    # img = cv2.imread('data/benchmark_maps/hou2_clean.png')
    img = cv2.imread('data/benchmark_maps/ryu.png')
    params = Parameter("config/ryu_params.yaml").params
    # params = Parameter("config/hou2_params.yaml").params

    ws, ws_erosion, dist_transform = segmentation.marker_controlled_watershed(img, params)
    bridge_nodes, bridge_edges = segmentation.find_bridge_nodes(ws_erosion, dist_transform)
    segment_envs, largest_rectangles = _create_rooms(ws_erosion, params)
    _create_paths(segment_envs, bridge_nodes, bridge_edges, params)
    _plot_all_envs(segment_envs)

    ws2 = segmentation.draw(ws, bridge_nodes, (22))
    # ws3 = segmentation.draw(ws2, largest_rectangles, (21))
    ws4 = _draw_all_paths(ws2, segment_envs, (25))
    segmentation.show_imgs(ws4, name="map_benchmark_ryu_result", save=False)
