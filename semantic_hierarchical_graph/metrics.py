import itertools
import json
from time import time
from typing import Dict, Any, List, Tuple
from collections import deque

import numpy as np
import cv2
from networkx.classes.function import path_weight
import semantic_hierarchical_graph.segmentation as segmentation
from semantic_hierarchical_graph.vector import Vector


class Metrics():
    def __init__(self, room) -> None:
        self.metrics: Dict[str, Any] = {}
        self.metrics.update(room.params)
        self.metrics["room_name"] = room.unique_name
        self.metrics["room_id"] = room.id
        self.metrics["num_nodes"] = len(room.get_childs())
        self.metrics["ILIR_path_length"] = sum([path.length for path in room.env.path])
        self.metrics["ILIR_avg_line_length"] = self.metrics["ILIR_path_length"] / len(room.env.path)

        path_metrics, room_mask_with_paths = self._calc_single_path_metrics(room)
        self.metrics.update(path_metrics)

        self.metrics["disturbance"] = self._calc_disturbance(room.mask, room_mask_with_paths)

        print("Bridge points not connected:", room.bridge_points_not_connected)
        # room_img = cv2.cvtColor(room.mask, cv2.COLOR_GRAY2RGB)
        # segmentation.show_imgs(room_img)

        # TODO: Deal with multiple possible solutions
        #       Check for correct distances on bridge connections

    def _calc_single_path_metrics(self, room) -> Tuple[Dict, np.ndarray]:
        room_mask: np.ndarray = room.mask.copy()
        path_metrics: Dict[str, Any] = {}
        path_metrics["success_rate"] = []
        path_metrics["planning_time"] = []
        path_metrics["path_length"] = []
        path_metrics["num_turns"] = []
        path_metrics["cumulative_turning_angle"] = []
        path_metrics["smoothness"] = []
        path_metrics["obstacle_clearance"] = []
        path_metrics["obstacle_clearance_min"] = np.inf

        bridge_points = [point for points in room.bridge_nodes.values() for point in points]
        path_metrics["num_bridge_points"] = len(bridge_points)

        # TODO: Generate more random points in the room to test
        # TODO: Implement function to plan from arbitrary point to arbitrary point. Needs shortest connection to existing roadmap

        for point_1, point_2 in itertools.combinations(bridge_points, 2):
            ts = time()
            path = room._plan(str(point_1), str(point_2))
            te = time()
            path_metrics["planning_time"].append(te - ts)
            if len(path) == 0:
                path_metrics["success_rate"].append(0)
                continue
            path_metrics["success_rate"].append(1)
            path_metrics["path_length"].append(self._calc_path_length(room.child_graph, path))
            turns, angles, smoothness = self._calc_smoothness(path)
            path_metrics["num_turns"].append(turns)
            path_metrics["cumulative_turning_angle"].append(angles)
            path_metrics["smoothness"].append(smoothness)
            clearance, clearance_min = self._calc_obstacle_clearance(room, path)
            path_metrics["obstacle_clearance"].append(clearance)
            path_metrics["obstacle_clearance_min"] = min(clearance_min, path_metrics["obstacle_clearance_min"])

            self._draw_path(room_mask, path, (0))
            # vis.draw_child_graph(room, path)

        return self._average_metrics(path_metrics), room_mask

    def _average_metrics(self, path_metrics: Dict) -> Dict:
        new_metrics = {}
        path_metrics["num_paths"] = len(path_metrics["success_rate"])
        for metric_name in path_metrics.keys():
            if isinstance(path_metrics[metric_name], list):
                if metric_name == "success_rate":
                    new_metrics[metric_name] = np.mean(path_metrics[metric_name])
                    continue
                if len(path_metrics[metric_name]) == 0:
                    continue
                new_metrics["avg_"+metric_name] = [np.mean(path_metrics[metric_name]),
                                                   np.std(path_metrics[metric_name]).item(),
                                                   np.min(path_metrics[metric_name]).item(),
                                                   np.max(path_metrics[metric_name]).item()]
        path_metrics.update(new_metrics)

        return path_metrics

    def _calc_path_length(self, graph, path) -> float:
        length = path_weight(graph, path, weight="distance")
        # dist = 0
        # for i in range(len(path) - 1):
        #     dist += util.get_euclidean_distance(path[i].pos, path[i + 1].pos)
        return length

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
        v0 = Vector.from_two_points(path[0].pos[:2], path[1].pos[:2])
        vectors.append(v0)
        length = v0.norm()

        for i in range(1, len(path)-1):
            vectors.append(Vector.from_two_points(path[i].pos[:2], path[i + 1].pos[:2]))
            angle = np.degrees(vectors[0].angle(vectors[1]))
            length += vectors[1].norm()
            if angle > 0:
                turns += 1
                angles += angle

        # smoothness = angles / turns
        normalized_smoothness = angles / length
        # print("Smoothness", smoothness, normalized_smoothness)

        return turns, angles, normalized_smoothness

    def _calc_obstacle_clearance(self, room, path) -> Tuple[float, float]:
        dist_transform = room.parent_node.dist_transform
        path_mask: np.ndarray = np.zeros(dist_transform.shape, dtype=np.uint8)
        self._draw_path(path_mask, path, 1)
        path_mask = np.where(path_mask == 1, True, False)
        path_distances = dist_transform[path_mask]
        # segmentation.show_imgs(path_mask)
        return np.mean(path_distances).item(), np.min(path_distances).item()

    def _calc_disturbance(self, room_mask, room_mask_with_paths: np.ndarray) -> float:
        # 0. remove not connected bridge nodes from list or try to plan and adjust success rate
        # 1. plan path between bridge nodes
        # 2. repeat for all combinations
        # 3. draw all paths on room mask with distinct color
        # 3.1. lookup all path pixel on dist_transform and calc metric obstacle_clearance
        # 3.2. change path to background color or threshold image
        # 4. get single segments with cv2.findContours
        # 5. calc max area and calc disturbance metric and draw in different color
        # 5.1 disturbance = Largest open area inside roadmap / Room area (- safety margin)
        # 6. convert colors to rgb
        # 7. Average metrics for all paths

        area = cv2.moments(room_mask, True)["m00"]
        contours, _ = cv2.findContours(room_mask_with_paths, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = list(map(cv2.contourArea, contours))

        # Only for visualization
        c_max = contours[np.argmax(contour_areas)]
        cv2.drawContours(room_mask_with_paths, [c_max], 0, (128), -1)
        segmentation.show_imgs(room_mask_with_paths)

        return max(contour_areas) / area

    def _draw_path(self, img: np.ndarray, path: List, color):
        for i in range(len(path) - 1):
            pt1 = np.round(path[i].pos[0:2]).astype("int32")
            pt2 = np.round(path[i + 1].pos[0:2]).astype("int32")
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_4)

    def print_metrics(self) -> None:
        print(self.metrics)

    def save_metrics(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)


if __name__ == "__main__":
    from semantic_hierarchical_graph.graph import SHGraph
    from semantic_hierarchical_graph.floor import Floor
    from semantic_hierarchical_graph import visualization as vis
    import semantic_hierarchical_graph.utils as util

    G = SHGraph(root_name="Benchmark", root_pos=(0, 0, 0))
    floor = Floor("ryu", G, (0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    room_14 = floor._get_child("room_14")

    metrics = Metrics(room_11)
    # metrics.print_metrics()
    metrics.save_metrics("data/ryu_metrics.json")

    # path = room_11._plan("(555, 211)", "(81, 358)")
    # path_list = util._map_names_to_nodes(path)

    # print(path_list)
    # vis.draw_child_graph_3d(room_11, path)

    # ws2 = segmentation.draw(floor.watershed, floor.all_bridge_nodes, (22))
    # ws4 = floor.draw_all_paths(ws2, (0), path_dict, (25))
    # segmentation.show_imgs(ws4, name="map_benchmark_ryu_result", save=False)
