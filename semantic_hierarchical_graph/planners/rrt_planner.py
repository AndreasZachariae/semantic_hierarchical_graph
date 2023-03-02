from typing import Any, List, Optional, Tuple
from path_planner_suite.IPEnvironment import CollisionChecker

from path_planner_suite.IPRRT import RRT
from semantic_hierarchical_graph.floor import Room
import semantic_hierarchical_graph.planners.planner_conversion as pc


class RRTPlanner():
    def __init__(self, room: Room, config: Optional[dict] = None):
        self.name = "RRT"
        self.room = room
        if config is None:
            self.config = dict()
            self.config["numberOfGeneratedNodes"] = 300
            self.config["testGoalAfterNumberOfNodes"] = 10
        else:
            self.config = config

        self.benchmark = pc.convert_room_to_IPBenchmark(room)
        self.planner = RRT(self.benchmark.collisionChecker)

    @classmethod
    def around_point(cls, point: Tuple[float, float], max_dist: float, scene: List, config: Optional[dict] = None):
        obj = cls.__new__(cls)
        obj.name = "RRT"
        obj.config = config
        IPlimits = [[round(point[0] - max_dist), round(point[0] + max_dist)],
                    [round(point[1] - max_dist), round(point[1] + max_dist)]]
        IPscene = {str(i): obstacle for i, obstacle in enumerate(scene)}
        IPcollision_checker = CollisionChecker(IPscene, IPlimits)
        obj.planner = RRT(IPcollision_checker)
        return obj

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
