from typing import List, Tuple
import cv2
import networkx as nx

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


def convert_path_to_PathNode(path, graph) -> List:
    pos = nx.get_node_attributes(graph, 'pos')
    return [PathNode(Position(pos[node][0], pos[node][1], 0.0)) for node in path]
