import itertools
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import cv2
from shapely.ops import nearest_points, unary_union, polygonize_full
from shapely import MultiLineString, Polygon, LineString, MultiPolygon, GeometryCollection, Point
import largestinteriorrectangle as lir
from semantic_hierarchical_graph.environment import Environment
import semantic_hierarchical_graph.planners.astar_planner as astar_planner
import semantic_hierarchical_graph.planners.rrt_planner as rrt_planner
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError
from semantic_hierarchical_graph.types.parameter import Parameter
import semantic_hierarchical_graph.segmentation as segmentation
from semantic_hierarchical_graph.types.position import Position


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
    is_corridor = [False]
    while True:
        segment = np.where(ws_erosion == env.room_id, 255, 0).astype("uint8")
        # TODO: Changed from cv2.RETR_TREE to EXTERNAL because it is faster and hierarchy doesn't matter
        contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_areas = list(map(cv2.contourArea, contours))
        c_max_index = np.argmax(contour_areas)
        c_max = contours[c_max_index]

        if first_loop:
            M = cv2.moments(segment)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = Position(cX, cY, 0)

            x, y, w, h = cv2.boundingRect(c_max)
            env.room_bbox = [x, y, w, h]
            env.floor_bbox = [0, 0, ws_erosion.shape[1], ws_erosion.shape[0]]
            padding = 10
            # interior polygon (= free room) has to be inverted with [::-1] to be clear space in shapely
            walls = Polygon([(x-1-padding, y-1-padding), (x-1-padding, y+h+padding),
                            (x+w+padding, y+h+padding), (x+w+padding, y-1-padding)], [c_max[:, 0, :][::-1]])
            env.add_obstacle(walls)
            obstacle_index = np.where(hierarchy[0, :, 3] == c_max_index)[0]
            [env.add_obstacle(Polygon(contours[index][:, 0, :])) for index in obstacle_index]

        elif np.max(contour_areas) < params["min_contour_area"]:
            # print("New contour too small", cv2.contourArea(c_max))
            break

        rectangle = lir.lir(segment.astype("bool"), c_max[:, 0, :])

        largest_rectangles.append(rectangle)

        path = _path_from_rectangle(rectangle, params, is_corridor, first_loop)
        if not path.is_empty:
            env.add_path(path)

        x, y, w, h = rectangle
        # Shrink rectangle by x pixel to make touching possible
        shrinkage = 1
        ws_erosion = cv2.rectangle(ws_erosion, (x+shrinkage, y+shrinkage),
                                   (x+w-1-shrinkage, y+h-1-shrinkage), (1), -1)
        first_loop = False

    # env.plot()
    # segmentation.show_imgs(segment, ws_erosion)
    return largest_rectangles, centroid


def _path_from_rectangle(rectangle: np.ndarray, params: dict, is_corridor: List, first_loop: bool) -> LineString:
    x, y, w, h = rectangle

    if (w * h) < params["min_roadmap_area"]:
        # print("area for roadmap too small")
        return LineString()

    # Offset of -1 pixel needed because opencv retuns w, h instead of coordinates
    # But shapely takes 1 pixel width edge as infinte thin line,
    # thus rectangles with edge 1 pixel different are not touching (not able to merge).
    # So rectangles are shrinked by 1 pixel in ws_tmp to compensate and make touching possible.

    w -= 1
    h -= 1
    if first_loop:
        if w < params["max_corridor_width"] or h < params["max_corridor_width"]:
            is_corridor[0] = True
            if w < h:
                point_1 = (x + w//2, y)
                point_2 = (x + w//2, y+h)
            else:
                point_1 = (x, y+h//2)
                point_2 = (x + w, y+h//2)

            # cv2.line(ws, point_1, point_2, (25), 2)
            return LineString([point_1, point_2])

    if not is_corridor[0]:
        point_1 = (x, y)
        point_2 = (x + w, y)
        point_3 = (x + w, y + h)
        point_4 = (x, y + h)
        # ws = cv2.rectangle(ws, rectangle, (25), 2)
        return LineString([point_1, point_2, point_3, point_4, point_1])
    else:
        return LineString()


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

    # Connect bridge points between each other if room has no roadmap
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

    # Merge rectangles to roadmap and connect bridge points to path
    else:
        print("Connecting paths in room", env.room_id)
        result, dangles, cuts, invalids = polygonize_full(env.path)
        union_polygon: Polygon = unary_union(result)
        if isinstance(union_polygon, MultiPolygon):
            max_poly = max(union_polygon.geoms, key=lambda x: x.area)
            env.path = [max_poly.boundary]
        elif isinstance(union_polygon, Polygon):
            env.path = [union_polygon.boundary]
        elif isinstance(union_polygon, GeometryCollection):
            # union polygon is empty. No polygon in original room path
            pass
        else:
            print("unknown shape returned from polygon union")
            raise SHGGeometryError()

        # First try to connect all points with a straight line
        bridge_points_not_connected_directly = []
        for point in bridge_points:
            connection, _ = env.find_shortest_connection(point)
            if connection is not None:
                env.add_path(connection)
            else:
                bridge_points_not_connected_directly.append(point)

        # Second try to connect remaining points with a planner
        for point in bridge_points_not_connected_directly:
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
    connection, closest_path = env.find_shortest_connection(point)
    if connection is None:
        path = _connect_point_with_astar(point, env, params)
        # path = _connect_point_with_rrt(point, env, params)

        if path == [] or path is None:
            print("No connection found for point", point)
            return [], None

        connections = [LineString([path[i].pos.xy, path[i+1].pos.xy]) for i in range(0, len(path)-1)]  # type: ignore
        connections.append(LineString([path[-1].pos.xy, point]))
        return connections, closest_path

    return [connection], closest_path


def _connect_point_with_rrt(point: Tuple[float, float], env: Environment, params: dict) -> List:
    config = dict()
    config["numberOfGeneratedNodes"] = 200
    config["testGoalAfterNumberOfNodes"] = 1
    config["smoothing_algorithm"] = "bechtold_glavina"
    config["smoothing_max_iterations"] = 100
    config["smoothing_max_k"] = 50
    planner = rrt_planner.RRTPlanner.around_point(point, params["max_distance_to_connect_points"], env.scene, config)

    pos = Point(point[0], point[1])
    closest_path = min(env.path, key=lambda x: x.distance(pos))
    closest_point: Point = nearest_points(closest_path, pos)[0]
    return planner.plan(point, (closest_point.x, closest_point.y, True))[0]  # type: ignore


def _connect_point_with_astar(point: Tuple[float, float], env: Environment, params: dict) -> List:
    import time
    config = dict()
    config["heuristic"] = 'euclidean'
    config["w"] = 0.5
    config['max_iterations'] = 10000
    config["smoothing_algorithm"] = "bechtold_glavina"
    config["smoothing_max_iterations"] = 100
    config["smoothing_max_k"] = 50
    planner = astar_planner.AStarPlanner.around_point(
        point, params["max_distance_to_connect_points"], env.scene, config)

    goal_list = _get_goal_points(env)
    ts = time.time()
    path = planner.plan_with_lists([[point[0], point[1]]], goal_list, True)[0]
    print("Time", time.time() - ts)
    return path  # type: ignore


def _get_goal_points(env: Environment):
    room_img = np.zeros((env.floor_bbox[3], env.floor_bbox[2]), np.uint8)
    room_img = _draw_path(room_img, env, (1,), 1)
    goal_points = np.flip(np.argwhere(room_img == 1))
    return goal_points.tolist()


def _draw_path(img: np.ndarray, env: Environment, color: Tuple, thickness: int) -> np.ndarray:
    for line in env.path:
        if isinstance(line, MultiLineString):
            for string in line.geoms:
                cv2.polylines(img, [string.coords._coords.astype("int32")], False,  color, thickness)
        else:
            cv2.polylines(img, [line.coords._coords.astype("int32")], False,  color, thickness)

    return img


def _plot_all_envs(envs: Dict[int, Environment]):
    all_envs = Environment(-1)
    for env in envs.values():
        [all_envs.add_obstacle(obstacle) for obstacle in env.scene]
        [all_envs.add_path(path) for path in env.path]

    all_envs.plot()


def _draw_all_paths(img: np.ndarray, envs: Dict[int, Environment],  color: Tuple) -> np.ndarray:
    img_new = img.copy()
    for env in envs.values():
        _draw_path(img_new, env, color, 2)

    return img_new


if __name__ == '__main__':

    # img = cv2.imread('data/benchmark_maps/hou2_clean.png')
    # img = cv2.imread('data/benchmark_maps/ryu.png')
    # params = Parameter("config/ryu_params.yaml").params
    # params = Parameter("config/hou2_params.yaml").params
    img = cv2.imread('data/graphs/simulation/floor/aws1.pgm')
    params = Parameter("data/graphs/simulation/floor/aws1.yaml").params

    ws, ws_erosion, dist_transform = segmentation.marker_controlled_watershed(img, params)
    bridge_nodes, bridge_edges = segmentation.find_bridge_nodes(ws_erosion, dist_transform)
    segment_envs, largest_rectangles = _create_rooms(ws_erosion, params)
    _create_paths(segment_envs, bridge_nodes, bridge_edges, params)
    _plot_all_envs(segment_envs)

    ws2 = segmentation.draw(ws, bridge_nodes, (44))
    # ws3 = segmentation.draw(ws2, largest_rectangles, (21))
    ws4 = _draw_all_paths(ws2, segment_envs, (44,))
    segmentation.show_imgs(ws4, name="map_benchmark_ryu_result", save=False)
