from typing import Tuple

from path_planner_suite.IPRRT import RRT
from semantic_hierarchical_graph.floor import Room
import semantic_hierarchical_graph.planners.planner_conversion as pc


class RRTPlanner():
    def __init__(self, room: Room):
        self.name = "RRT"
        self.room = room
        self.config = dict()
        self.config["numberOfGeneratedNodes"] = 300
        self.config["testGoalAfterNumberOfNodes"] = 10

        self.benchmark = pc.convert_room_to_IPBenchmark(room)
        self.planner = RRT(self.benchmark.collisionChecker)

    def plan(self, start: Tuple, goal: Tuple):
        start_list, goal_list = pc.convert_to_start_goal_lists(start, goal)
        try:
            path = self.planner.planPath(start_list, goal_list, self.config)
        except Exception as e:
            print("Error while planning with RRTPlanner: ")
            print(e)
            path = None

        if path is not None:
            path = pc.convert_path_to_PathNode(path, self.planner.graph)
        return path, self.planner.graph

    def plan_in_map_frame(self, start: Tuple, goal: Tuple):
        start_pos, goal_pos = pc.convert_map_frame_to_grid(start, goal, self.room.params["grid_size"])
        self.plan(start_pos, goal_pos)
