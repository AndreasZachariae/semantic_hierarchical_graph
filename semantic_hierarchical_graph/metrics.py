import json
from typing import Dict, Any
# from semantic_hierarchical_graph.floor import Room


class Metrics():
    def __init__(self, room) -> None:
        self.metrics: Dict[str, Any] = {}
        self.metrics["name"] = room.unique_name
        self.metrics["id"] = room.id
        self.metrics.update(room.params)
        self.metrics["num_nodes"] = len(room.get_childs())
        self.metrics["path_length"] = sum([path.length for path in room.env.path])
        self.metrics["average_line_length"] = self.metrics["path_length"] / len(room.env.path)
        self.metrics["num_turns"] = 0  # define comparable start end end points
        self.metrics["cumulative_turning_angle"] = 0
        self.metrics["smoothness"] = self._calc_smoothness(room.env)
        self.metrics["planning_time"] = 0
        self.metrics["public_disturbance"] = self._calc_public_disturbance(room.env)
        self.metrics["success rate"] = 0 # over multiple runs. For PRB between same goals or for Deterministic between all points of a floor
        self.metrics["obstacle_clearance"] = self._calc_obstacle_clearance(room.env)

    def _calc_smoothness(self, env) -> float:
        smoothness = 0
        # As defined in this paper https://www.mdpi.com/1424-8220/20/23/6822
        # Sum up all path segemnts and take the min angle between the segment and the following segment.
        # 180° is the best case, 0° is the worst case.
        # Turn path into vector and calculate angle between vectors.

        # Normalize to path length
        # Rectangles with 90 degree turns always have the same smoothness
        # Sum up all turning angles and divide by the number of turns or divied by the path length
        
        # Alternative paper https://arxiv.org/pdf/2203.03092.pdf
        # Smoothness of trajectory (degrees). The average angle change between consecutive segments of paths shows how drastic and sudden the agent’s movement changes could be

        # Alternative paper
        # The path smoothness can be calculated by using the following formula:
#         S(P) = XDi=1αi =XDi=1 arccos((Pi − Pi−1) · (Pi+1 − Pi)|Pi − Pi−1| × |Pi+1 − Pi| × 180)
#         where αi refers to the value of the i-th deflection angle of the generated path (measured in radians in the
#         range from 0 to π). (Pi − Pi−1) · (Pi+1 − Pi) indicates the inner product between vectors of Pi − Pi−1
#         and Pi+1 − Pi while |Pi − Pi−1| denotes the vector norm.
        return smoothness

    def _calc_public_disturbance(self, env) -> float:
        disturbance = 0

        # Largest open area inside roadmap / Room area (- safety margin)
        return disturbance

    def print_metrics(self) -> None:
        print(self.metrics)

    def save_metrics(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)
