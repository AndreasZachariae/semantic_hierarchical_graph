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
        self.metrics["average_line_length"] = self._calc_average_line_length(room.env)
        self.metrics["num_turns"] = 0  # define comparable start end end points
        self.metrics["cumulative_turning_angle"] = 0
        self.metrics["smoothness"] = self._calc_smoothness(room.env)
        self.metrics["planning_time"] = 0
        self.metrics["public_disturbance"] = self._calc_public_disturbance(room.env)

    def _calc_average_line_length(self, env) -> float:
        return sum([path.length for path in env.path]) / len(env.path)

    def _calc_smoothness(self, env) -> float:
        smoothness = 0
        # As defined in this paper https://www.mdpi.com/1424-8220/20/23/6822
        # Sum up all path segemnts and take the min angle between the segment and the following segment.
        # 180° is the best case, 0° is the worst case.
        # Turn path into vector and calculate angle between vectors.

        # Normalize to path length
        # Rectangles with 90 degree turns always have the same smoothness
        # Sum up all turning angles and divide by the number of turns or divied by the path length

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
