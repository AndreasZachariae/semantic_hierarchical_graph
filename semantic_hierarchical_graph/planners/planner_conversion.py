from typing import List, Tuple
import cv2

from path_planner_suite.IPBenchmark import Benchmark
from path_planner_suite.IPEnvironment import CollisionChecker
from semantic_hierarchical_graph.floor import Room
from semantic_hierarchical_graph.types.position import Position


class PathNode():
    def __init__(self, pos: Position):
        self.pos: Position = pos
        self.pos_abs: Position = pos
        self.unique_name: str = pos.to_name()


def convert_to_start_goal_lists(start: Tuple, goal: Tuple) -> Tuple[List, List]:
    start_list = [round(start[0]), round(start[1])]
    goal_list = [round(goal[0]), round(goal[1])]
    return [start_list], [goal_list]


def convert_map_frame_to_grid(start: Tuple, goal: Tuple, grid_size: float) -> Tuple[Tuple, Tuple]:
    start_pos = Position.convert_to_grid(start, grid_size)
    goal_pos = Position.convert_to_grid(goal, grid_size)
    return start_pos.xy, goal_pos.xy


def convert_room_to_IPBenchmark(room: Room):
    box = cv2.boundingRect(room.mask)
    IPlimits = [[box[0], box[0] + box[2]], [box[1], box[1] + box[3]]]
    IPscene = {str(i): obstacle for i, obstacle in enumerate(room.env.scene)}
    IPcollision_checker = CollisionChecker(IPscene, IPlimits)
    return Benchmark(f"room_{room.id}", IPcollision_checker, [], [], "", 1)


def convert_path_to_PathNode(path):
    return [PathNode(_get_pos_from_node_id(node)) for node in path]


def _get_pos_from_node_id(node_id):
    pos_split = node_id.split("-")
    pos = Position(float(pos_split[1]), float(pos_split[2]), 0.0)
    return pos
