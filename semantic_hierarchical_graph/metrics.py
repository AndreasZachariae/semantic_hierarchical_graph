import itertools
import json
from time import time
from typing import Dict, Any, List, Tuple
from collections import deque
from math import comb
import numpy as np
import cv2
from shapely import Point

import semantic_hierarchical_graph.segmentation as segmentation
from semantic_hierarchical_graph.types.vector import Vector
from semantic_hierarchical_graph.types.position import Position
from semantic_hierarchical_graph.floor import Room
from semantic_hierarchical_graph.planners.astar_planner import AStarPlanner
from semantic_hierarchical_graph.planners.ilir_planner import ILIRPlanner
from semantic_hierarchical_graph.planners.prm_planner import PRMPlanner
from semantic_hierarchical_graph.planners.rrt_planner import RRTPlanner


class Metrics():
    def __init__(self, room: Room) -> None:
        self.metrics: Dict[str, Any] = {}
        self.metrics.update(room.params)
        self.metrics["room_name"] = room.unique_name
        self.metrics["room_id"] = room.id
        self.metrics["num_nodes"] = len(room.child_graph)
        self.metrics["ILIR_path_length"] = sum([path.length for path in room.env.path])
        self.metrics["ILIR_avg_line_length"] = self.metrics["ILIR_path_length"] / len(room.env.path)

        bridge_points = [point for points in room.bridge_nodes.values() for point in points]
        self.metrics["num_bridge_points"] = len(bridge_points)

        # random_points = self._get_random_valid_points(room, n=10)
        # Room 2
        random_points = [(380, 72), (363, 63), (325, 43), (276, 129), (302, 170),
                         (273, 45), (373, 193), (342, 161), (393, 43), (339, 76)]
        # Room 11
        # random_points = [(75, 275), (554, 341), (611, 287), (509, 283), (198, 296),
        #                  (484, 300), (440, 303), (446, 314), (480, 265), (556, 296)]
        # Hou2 room 9
        # random_points = [(280, 433), (435, 230), (171, 276), (297, 456), (280, 448),
        #                  (153, 222), (111, 224), (283, 399), (452, 264), (300, 366)]

        self.metrics["num_random_points"] = len(random_points)
        print("Random points:", random_points)
        bridge_points.extend(random_points)
        self.metrics["num_paths"] = comb(len(bridge_points), 2)

        # AStarPlanner(room), ILIRPlanner(room), PRMPlanner(room), RRTPlanner(room)
        for planner in [AStarPlanner(room), ILIRPlanner(room), PRMPlanner(room), RRTPlanner(room),]:
            path_metrics, room_mask_with_paths = self._calc_single_path_metrics(room, bridge_points, planner)
            self.metrics[planner.name] = planner.config
            path_metrics["disturbance"] = self._calc_disturbance(room.mask, room_mask_with_paths)
            self.metrics[planner.name].update(path_metrics)

        print("Bridge points not connected:", room.bridge_points_not_connected)
        # room_img = cv2.cvtColor(room.mask, cv2.COLOR_GRAY2RGB)
        # segmentation.show_imgs(room_img)

        # TODO: Deal with multiple possible solutions
        #       Check for correct distances on bridge connections

    def _calc_single_path_metrics(self, room: Room, points: List, planner) -> Tuple[Dict, np.ndarray]:
        room_mask: np.ndarray = room.mask.copy()
        path_metrics: Dict[str, Any] = {}
        path_metrics["success_rate"] = []
        path_metrics["planning_time"] = []
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
            path, vis_graph = planner.plan(point_1, point_2, True)
            te = time()
            path_metrics["planning_time"].append(te - ts)
            if path is None or path == []:
                path_metrics["success_rate"].append(0)
                print(f"No path found from {point_1} to {point_2}")
                continue
            path_metrics["success_rate"].append(1)
            path_metrics["path_length"].append(self._calc_path_length(vis_graph, path))
            turns, angles, smoothness = self._calc_smoothness(path)
            path_metrics["num_turns"].append(turns)
            path_metrics["cumulative_turning_angle"].append(angles)
            path_metrics["smoothness"].append(smoothness)
            clearance, clearance_min, clearance_std = self._calc_obstacle_clearance(room, path)
            path_metrics["obstacle_distance_std"].append(clearance_std)
            path_metrics["obstacle_clearance"].append(clearance)
            path_metrics["obstacle_clearance_min"] = min(clearance_min, path_metrics["obstacle_clearance_min"])
            path_metrics["centroid_distance"].append(
                self._calc_centroid_distance(room.centroid, room.mask.copy(),  path))

            self._draw_path(room_mask, path, (0))
            # segmentation.show_imgs(room_mask)
            # vis.draw_child_graph(room, path, vis_graph)

        return self._average_metrics(path_metrics), room_mask

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

    def _get_random_valid_points(self, room: Room, n: int) -> List[Tuple]:
        points = []
        box = cv2.boundingRect(room.mask)
        while len(points) < n:
            x = np.random.randint(box[0], box[0] + box[2])
            y = np.random.randint(box[1], box[1] + box[3])
            if not room.env._in_collision(Point(x, y)):
                points.append((x, y))

        return points

    def _calc_path_length(self, graph, path) -> float:
        dist = 0
        for i in range(len(path) - 1):
            dist += path[i].pos.distance(path[i + 1].pos)
        return dist

    def _calc_smoothness(self, path) -> Tuple[int, float, float]:
        turns, angles, smoothness = 0, 0, 0
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

    def _calc_obstacle_clearance(self, room: Room, path) -> Tuple[float, float, float]:
        dist_transform = room.parent_node.dist_transform  # type: ignore
        path_mask: np.ndarray = np.zeros(dist_transform.shape, dtype=np.uint8)
        self._draw_path(path_mask, path, 1)
        path_mask = np.where(path_mask == 1, True, False)
        path_distances = dist_transform[path_mask]
        # segmentation.show_imgs(path_mask)
        return np.mean(path_distances).item(), np.min(path_distances).item(), np.std(path_distances).item()

    def _calc_centroid_distance(self, centroid: Position, room_mask: np.ndarray, path) -> float:
        self._draw_path(room_mask, path, 1)
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
        # segmentation.show_imgs(labels)
        segmentation.show_imgs(
            room_mask_with_paths, name=f"{self.metrics['room_name']}_disturbance_{list(self.metrics.keys())[-1]}", save=False)

        return 1 - (largest_free_area / area)

    def _draw_path(self, img: np.ndarray, path: List, color):
        for i in range(len(path) - 1):
            pt1 = np.round(path[i].pos.xy).astype("int32")
            pt2 = np.round(path[i + 1].pos.xy).astype("int32")
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_4)

    def print_metrics(self) -> None:
        print(self.metrics)

    def save_metrics(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)


if __name__ == "__main__":
    from semantic_hierarchical_graph.graph import SHGraph
    from semantic_hierarchical_graph.floor import Floor

    G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    # floor = Floor("hou2", G, Position(0, 0, 1), 'data/benchmark_maps/hou2_clean.png', "config/hou2_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    room = floor._get_child("room_2")
    # room = floor._get_child("room_11")
    # room = floor._get_child("room_9")

    metrics = Metrics(room)
    # metrics.print_metrics()
    # metrics.save_metrics("data/hou2_metrics.json")
    metrics.save_metrics("data/ryu_metrics.json")
