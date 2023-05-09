import itertools
import json
from time import time
from typing import Dict, Any, List, Tuple
from collections import deque
from math import comb
import numpy as np
import cv2
from shapely import Point
import sys

import semantic_hierarchical_graph.segmentation as segmentation
from semantic_hierarchical_graph.types.vector import Vector
from semantic_hierarchical_graph.types.position import Position
from semantic_hierarchical_graph.floor import Room, Floor
from semantic_hierarchical_graph.planners.astar_planner import AStarPlanner
from semantic_hierarchical_graph.planners.ilir_planner import ILIRPlanner
from semantic_hierarchical_graph.planners.prm_planner import PRMPlanner
from semantic_hierarchical_graph.planners.rrt_planner import RRTPlanner
from semantic_hierarchical_graph.planners.shg_planner import SHGPlanner
from semantic_hierarchical_graph.metrics_plots import plot_metrics
import semantic_hierarchical_graph.utils as utils

print("Recursion limit increased from", sys.getrecursionlimit(), "to", 2000)
sys.setrecursionlimit(2000)


class Metrics():
    def __init__(self, floor: Floor, graph_path: str) -> None:
        self.metrics: Dict[str, Any] = {}
        self.metrics.update(floor.params)
        self.metrics["floor_name"] = floor.unique_name
        self.metrics["num_rooms"] = len(floor.child_graph)

        bridge_points = [point for points in floor.all_bridge_nodes.values() for point in points]
        self.metrics["num_bridge_points"] = len(bridge_points)

        random_points = [(1219, 253), (1090, 500), (674, 277), (487, 412), (621, 263), (484, 95), (508, 131), (100, 500), (1209, 483), (
            1373, 177), (784, 54), (554, 189), (804, 149), (812, 57), (1510, 379), (350, 45), (258, 255), (511, 383), (535, 193), (1140, 261)]

        # random_points = self._get_random_valid_points(floor, n=20)
        self.metrics["num_random_points"] = len(random_points)
        print("Random points:", random_points)
        bridge_points.extend(random_points)
        self.metrics["num_paths"] = comb(len(bridge_points), 2)

        prm_config = {"radius": 50, "numNodes": 3000,
                      "smoothing_algorithm": "random", "smoothing_max_iterations": 100, "smoothing_max_k": 50}
        rrt_config = {"numberOfGeneratedNodes": 3000, "testGoalAfterNumberOfNodes": 100,
                      "smoothing_algorithm": "random", "smoothing_max_iterations": 100, "smoothing_max_k": 50}
        astar_config = {"heuristic": 'euclidean', "w": 0.5, 'max_iterations': 1000000,
                        "smoothing_algorithm": "random", "smoothing_max_iterations": 100, "smoothing_max_k": 50}

        # PRMPlanner(floor, prm_config), RRTPlanner(floor, rrt_config), AStarPlanner(floor, astar_config), SHGPlanner(graph_path)
        for planner in [PRMPlanner(floor, prm_config)]:
            path_metrics, room_mask_with_paths = self._calc_single_path_metrics(floor, bridge_points, planner)
            self.metrics[planner.name] = planner.config
            # TODO: Adapt to floor
            # path_metrics["disturbance"] = self._calc_disturbance(floor.mask, room_mask_with_paths)
            segmentation.show_imgs(room_mask_with_paths,
                                   name=f"{self.metrics['floor_name']}_{list(self.metrics.keys())[-1]}",
                                   save=False)
            self.metrics[planner.name].update(path_metrics)

        # TODO: exclude bridge points not connected
        print("Bridge points not connected:", floor.bridge_points_not_connected)
        # room_img = cv2.cvtColor(room_mask_with_paths, cv2.COLOR_GRAY2RGB)

    def _calc_single_path_metrics(self, floor: Floor, points: List, planner) -> Tuple[Dict, np.ndarray]:
        floor_mask: np.ndarray = floor.watershed.copy()
        path_metrics: Dict[str, Any] = {}
        path_metrics["success_rate"] = []
        path_metrics["planning_time"] = []
        path_metrics["smoothing_time"] = []
        path_metrics["path_length"] = []
        path_metrics["num_turns"] = []
        path_metrics["cumulative_turning_angle"] = []
        path_metrics["smoothness"] = []
        path_metrics["obstacle_distance_std"] = []
        path_metrics["obstacle_clearance"] = []
        path_metrics["obstacle_clearance_min"] = np.inf
        path_metrics["centroid_distance"] = []

        for point_1, point_2 in itertools.combinations(points, 2):
            print(f"Planning path {len(path_metrics['success_rate'])+1}/{self.metrics['num_paths']}")
            ts = time()
            path, vis_graph, smoothing_time = planner.plan_on_floor(floor.unique_name, point_1, point_2, True)
            te = time()
            path_metrics["planning_time"].append(te - ts)
            path_metrics["smoothing_time"].append(smoothing_time)
            if path is None or path == []:
                path_metrics["success_rate"].append(0)
                print(f"No path found from {point_1} to {point_2}")
                continue
            path_metrics["success_rate"].append(1)
            path_metrics["path_length"].append(self._calc_path_length(path))
            turns, angles, smoothness = self._calc_smoothness(path)
            path_metrics["num_turns"].append(turns)
            path_metrics["cumulative_turning_angle"].append(angles)
            path_metrics["smoothness"].append(smoothness)
            clearance, clearance_min, clearance_std = self._calc_obstacle_clearance(floor, path)
            path_metrics["obstacle_distance_std"].append(clearance_std)
            path_metrics["obstacle_clearance"].append(clearance)
            path_metrics["obstacle_clearance_min"] = min(clearance_min, path_metrics["obstacle_clearance_min"])
            # TODO: Adapt to floor
            # path_metrics["centroid_distance"].append(
            #     self._calc_centroid_distance(floor.centroid, floor.mask.copy(),  path))

            self._draw_path(floor_mask, path, 1, (22))
            # segmentation.show_imgs(floor_mask)
            # vis.draw_child_graph(room, path, vis_graph)

            # print(f"Dict size: {utils.get_obj_size(path_metrics) / 1024 / 1024} MB")
            # print(f"Path size: {utils.get_obj_size(path) / 1024 / 1024} MB")
            # print(f"Floor size: {utils.get_obj_size(floor) / 1024 / 1024} MB")
            # print(f"Graph size: {utils.get_obj_size(vis_graph) / 1024 / 1024} MB")
            # print(f"Planner size: {utils.get_obj_size(planner) / 1024 / 1024} MB")
            del path
            del vis_graph

        return self._average_metrics(path_metrics), floor_mask

    def _average_metrics(self, path_metrics: Dict) -> Dict:
        new_metrics = {}
        for metric_name in path_metrics.keys():
            if isinstance(path_metrics[metric_name], list):
                if metric_name == "success_rate":
                    new_metrics[metric_name] = np.mean(path_metrics[metric_name])
                    continue
                if len(path_metrics[metric_name]) == 0:
                    continue
                # new_metrics["avg_"+metric_name] =
                new_metrics[metric_name] = {"mean": np.mean(path_metrics[metric_name]),
                                            "std": np.std(path_metrics[metric_name]).item(),
                                            "min": np.min(path_metrics[metric_name]).item(),
                                            "max": np.max(path_metrics[metric_name]).item()}

        path_metrics.update(new_metrics)

        return path_metrics

    def _get_random_valid_points(self, floor: Floor, n: int) -> List[Tuple]:
        points = []
        shape = floor.ws_erosion.shape
        while len(points) < n:
            x = np.random.randint(0, shape[1])
            y = np.random.randint(0, shape[0])
            if int(floor.ws_erosion[y, x]) not in [0, -1, 1]:
                points.append((x, y))

        return points

    def _calc_path_length(self, path) -> float:
        dist = 0
        for i in range(len(path) - 1):
            dist += path[i].pos.distance(path[i + 1].pos)
        return dist

    def _calc_smoothness(self, path) -> Tuple[int, float, float]:
        turns, angles = 0, 0
        # As defined in this paper https://www.mdpi.com/1424-8220/20/23/6822
        # Sum up all path segemnts and take the min angle between the segment and the following segment.
        # 180° is the best case, 0° is the worst case.
        # Turn path into vector and calculate angle between vectors.

        # Normalize to path length
        # Rectangles with 90 degree turns always have the same smoothness
        # Sum up all turning angles and divide by the number of turns or divied by the path length

        # Alternative paper https://arxiv.org/pdf/2203.03092.pdf
        # Smoothness of trajectory (degrees). The average angle change between consecutive segments of paths shows how drastic and sudden the agent’s movement changes could be

        # Alternative paper https://doi.org/10.1007/s11432-016-9115-2
        # The path smoothness can be calculated by using the following formula:
        # S(P) = XDi=1αi =XDi=1 arccos((Pi − Pi−1) · (Pi+1 − Pi)|Pi − Pi−1| × |Pi+1 − Pi| × 180)
        # where αi refers to the value of the i-th deflection angle of the generated path (measured in radians in the
        # range from 0 to π). (Pi − Pi−1) · (Pi+1 − Pi) indicates the inner product between vectors of Pi − Pi−1
        # and Pi+1 − Pi while |Pi − Pi−1| denotes the vector norm.

        # Alterntive von Philipp
        # 1. Anzahl an nulldurchgängen, das heißt eine langer bogen mit mehreren winkeländerungen in die selbe richtung zählt als 1
        # 2. Gewichtete Anzahl der turns mit stärke des Winkels 180° = 1, 0° = 0 und damit die jeden gewichteten turn aufsummieren

        vectors = deque(maxlen=2)
        v0 = Vector.from_two_points(path[0].pos.xy, path[1].pos.xy)
        vectors.append(v0)
        length = v0.norm()

        for i in range(1, len(path)-1):
            vectors.append(Vector.from_two_points(path[i].pos.xy, path[i + 1].pos.xy))
            angle = np.degrees(vectors[0].angle(vectors[1]))
            length += vectors[1].norm()
            if angle > 0:
                turns += 1
                angles += angle

        # smoothness = angles / turns
        normalized_smoothness = angles / length
        # print("Smoothness", smoothness, normalized_smoothness)

        return turns, angles, normalized_smoothness

    def _calc_obstacle_clearance(self, floor: Floor, path) -> Tuple[float, float, float]:
        dist_transform = floor.dist_transform  # type: ignore
        path_mask: np.ndarray = np.zeros(dist_transform.shape, dtype=np.uint8)
        self._draw_path(path_mask, path, 1, 1)
        path_mask = np.where(path_mask == 1, True, False)
        path_distances = dist_transform[path_mask]
        # segmentation.show_imgs(path_mask)
        return np.mean(path_distances).item(), np.min(path_distances).item(), np.std(path_distances).item()

    def _calc_centroid_distance(self, centroid: Position, room_mask: np.ndarray, path) -> float:
        self._draw_path(room_mask, path, 1, 1)
        path_points = np.argwhere(room_mask == 1)
        distances = [((centroid.x - point[1]) ** 2 + (centroid.y - point[0]) ** 2)**0.5 for point in path_points]
        return np.mean(distances).item()

    def _calc_disturbance(self, room_mask, room_mask_with_paths: np.ndarray) -> float:
        # 0. remove not connected bridge nodes from list or try to plan and adjust success rate
        # 1. plan path between bridge nodes
        # 2. repeat for all combinations
        # 3. draw all paths on room mask with distinct color
        # 3.1. lookup all path pixel on dist_transform and calc metric obstacle_clearance
        # 3.2. change path to background color or threshold image
        # 4. get single segments with connectedComponentsWithStats
        # 5. calc max area and calc disturbance metric and draw in different color
        # 5.1 disturbance = Largest open area inside roadmap / Room area (- safety margin)
        # 6. convert colors to rgb
        # 7. Average metrics for all paths

        area = cv2.moments(room_mask, True)["m00"]
        _, labels, stats, _ = cv2.connectedComponentsWithStats(room_mask_with_paths, connectivity=4)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_free_area = stats[largest_label, cv2.CC_STAT_AREA]
        # print("Largest free area", largest_label, largest_free_area, stats)

        # Only for visualization
        room_mask_with_paths[np.where(labels == largest_label)] = 128
        # segmentation.show_imgs(
        #     room_mask_with_paths, name=f"{self.metrics['room_name']}_disturbance_{list(self.metrics.keys())[-1]}", save=False)

        return 1 - (largest_free_area / area)

    def _draw_path(self, img: np.ndarray, path: List, thickness, color):
        for i in range(len(path) - 1):
            pt1 = np.round(path[i].pos.xy).astype("int32")
            pt2 = np.round(path[i + 1].pos.xy).astype("int32")
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_4)

    def print_metrics(self) -> None:
        print(self.metrics)

    def save_metrics(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)


if __name__ == "__main__":
    from semantic_hierarchical_graph.graph import SHGraph

    # G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    # floor_ryu = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    # floor_hou2 = Floor("hou2", G, Position(0, 0, 1), 'data/benchmark_maps/hou2_clean.png', "config/hou2_params.yaml")
    # G.add_child_by_node(floor_ryu)
    # G.add_child_by_node(floor_hou2)
    # floor_ryu.create_rooms()
    # floor_ryu.create_bridges()
    # floor_hou2.create_rooms()
    # floor_hou2.create_bridges()

    # G.save_graph("data/tmp/graph.pickle")
    G = SHGraph.load_graph("data/tmp/graph.pickle")
    print(G.get_childs("name"))

    metrics = Metrics(G._get_child("ryu"), "data/tmp")
    # metrics.print_metrics()
    # metrics.save_metrics("data/floor_hou2_metrics.json")
    metrics.save_metrics("data/tmp/floor_ryu_metrics.json")

    # plot_metrics(metrics.metrics, "data/tmp/floor_ryu_metrics.png")
