from typing import List, Optional, Tuple

from path_planner_suite.IPAStar import AStar
from path_planner_suite.IPEnvironment import CollisionChecker
from path_planner_suite.IPSmoothing import IPSmoothing
from semantic_hierarchical_graph.floor import Room
import semantic_hierarchical_graph.planners.planner_conversion as pc


class AStarPlanner():
    def __init__(self, room: Room, config: Optional[dict] = None):
        self.name = "AStar"
        self.room = room
        if config is None:
            self.config = dict()
            self.config["heuristic"] = 'euclidean'
            self.config["w"] = 0.5
            self.config["smoothing_enabled"] = True
            self.config["smoothing_iterations"] = 50
            self.config["smoothing_max_k"] = 20
            self.config["smoothing_epsilon"] = 0.5
            self.config["smoothing_variance_window"] = 10
            self.config["smoothing_min_variance"] = 0.0
        else:
            self.config = config

        self.collision_checker = pc.convert_room_to_IPCollisionChecker(room)
        self.planner = AStar(self.collision_checker)

    @classmethod
    def around_point(cls, point: Tuple[float, float], max_dist: float, scene: List, config: Optional[dict] = None):
        obj = cls.__new__(cls)
        obj.name = "AStar"
        obj.config = config
        IPlimits = [[round(point[0] - max_dist), round(point[0] + max_dist)],
                    [round(point[1] - max_dist), round(point[1] + max_dist)]]
        IPscene = {str(i): obstacle for i, obstacle in enumerate(scene)}
        obj.collision_checker = CollisionChecker(IPscene, IPlimits)
        obj.planner = AStar(obj.collision_checker)
        return obj

    def plan(self, start: Tuple, goal: Tuple):
        start_list, goal_list = pc.convert_to_start_goal_lists(start, goal)
        try:
            path = self.planner.planPath(start_list, goal_list, self.config)
            print(len(path))
        except Exception as e:
            print("Error while planning with AStarPlanner: ")
            print(e)
            return None, self.planner.graph

        if self.config["smoothing_enabled"]:
            smoother = IPSmoothing(self.planner.graph, path, self.collision_checker)
            path = smoother.smooth_solution(self.config["smoothing_iterations"],
                                            self.config["smoothing_max_k"],
                                            self.config["smoothing_epsilon"],
                                            self.config["smoothing_variance_window"],
                                            self.config["smoothing_min_variance"])
            print(len(path), "smoothing finished")

        path = pc.convert_path_to_PathNode(path, self.planner.graph)
        print(len(path), "planning finished")

        return path, self.planner.graph

    def plan_in_map_frame(self, start: Tuple, goal: Tuple):
        start_pos, goal_pos = pc.convert_map_frame_to_grid(start, goal, self.room.params["grid_size"])
        self.plan(start_pos, goal_pos)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from semantic_hierarchical_graph.graph import SHGraph
    from semantic_hierarchical_graph.floor import Room, Floor
    import semantic_hierarchical_graph.visualization as vis
    import semantic_hierarchical_graph.utils as util
    from semantic_hierarchical_graph.types.position import Position
    G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    room_14 = floor._get_child("room_14")
    planner = AStarPlanner(room_11)
    path, vis_graph = planner.plan((480, 250), (75, 260))  # type: ignore
    # path, _ = planner.plan((555, 211), (81, 358))
    path_list = util._map_names_to_nodes(path)

    print(path_list)
    print("Path length", len(path_list))
    img = room_11.mask
    color = (128)
    path: List = path
    for i in range(len(path) - 1):
        pt1 = np.round(path[i].pos.xy).astype("int32")
        pt2 = np.round(path[i + 1].pos.xy).astype("int32")
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_4)
    plt.imshow(img)
    plt.show()
