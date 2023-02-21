from typing import Tuple

import cv2

from path_planner_suite.IPAStar import AStar
from path_planner_suite.IPBenchmark import Benchmark
from path_planner_suite.IPEnvironment import CollisionChecker
from semantic_hierarchical_graph.floor import Room
from semantic_hierarchical_graph.types import Position


class AStarPlanner():
    def __init__(self, room: Room):
        self.name = "AStar"
        self.room = room
        self.config = dict()
        self.config["heuristic"] = 'euclidean'
        self.config["w"] = 0.5

        self.benchmark = self._convert_room_to_benchmark(room)
        self.planner = AStar(self.benchmark.collisionChecker)

    def plan(self, start: Tuple, goal: Tuple):
        start_list = [round(start[0]), round(start[1])]
        goal_list = [round(goal[0]), round(goal[1])]
        path = self.planner.planPath([start_list], [goal_list], self.config)

        return path, self.planner.graph

    def plan_in_map_frame(self, start: Tuple, goal: Tuple):
        start_pos = Position.convert_to_grid(start, self.room.params["grid_size"])
        goal_pos = Position.convert_to_grid(goal, self.room.params["grid_size"])
        self.plan(start_pos.xy, goal_pos.xy)

    def _convert_room_to_benchmark(self, room: Room):
        box = cv2.boundingRect(room.mask)
        IPlimits = [[box[0], box[1]], box[0] + box[2], box[1] + box[3]]
        IPscene = {str(i): obstacle for i, obstacle in enumerate(room.env.scene)}
        IPcollision_checker = CollisionChecker(IPscene, IPlimits)
        return Benchmark(f"room_{room.id}", IPcollision_checker, [], [], "", 1)
