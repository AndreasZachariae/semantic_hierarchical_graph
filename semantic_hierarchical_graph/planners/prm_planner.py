from typing import Optional, Tuple

from path_planner_suite.IPBasicPRM import BasicPRM
import semantic_hierarchical_graph.planners.planner_conversion as pc


class PRMPlanner():
    def __init__(self, room, config: Optional[dict] = None):
        self.name = "PRM"
        self.room = room
        if config is None:
            self.config = dict()
            self.config["radius"] = 50
            self.config["numNodes"] = 300
        else:
            self.config = config

        self.collision_checker = pc.convert_room_to_IPCollisionChecker(room)
        self.planner = BasicPRM(self.collision_checker)

    def plan(self, start: Tuple, goal: Tuple):
        start_list, goal_list = pc.convert_to_start_goal_lists(start, goal)
        try:
            path = self.planner.planPath(start_list, goal_list, self.config)
        except Exception as e:
            print("Error while planning with PRMPlanner: ")
            print(e)
            path = None

        if path is not None:
            path = pc.convert_path_to_PathNode(path, self.planner.graph)
        return path, self.planner.graph

    def plan_in_map_frame(self, start: Tuple, goal: Tuple):
        start_pos, goal_pos = pc.convert_map_frame_to_grid(start, goal, self.room.params["grid_size"])
        self.plan(start_pos, goal_pos)
