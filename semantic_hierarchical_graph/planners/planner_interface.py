from typing import Any, Dict, List, Tuple
import cv2
import networkx as nx

from path_planner_suite.IPEnvironment import CollisionChecker
from path_planner_suite.IPSmoothing import IPSmoothing
from semantic_hierarchical_graph.types.position import Position


class PathNode():
    def __init__(self, pos: Position):
        self.pos: Position = pos
        self.pos_abs: Position = pos
        self.unique_name: str = pos.to_name()

    def __str__(self):
        return self.unique_name


class PlannerInterface():
    def __init__(self, room, config: Dict):
        # Do not create a new instance of this Interface directly, use one of the subclasses
        self.room = room
        self.config = config
        self.collision_checker = self._convert_room_to_IPCollisionChecker(room)
        self.planner: Any
        self.name: str

    def plan(self, start: Tuple, goal: Tuple, smoothing_enabled: bool):
        start_list, goal_list = self._convert_to_start_goal_lists(start, goal)
        return self.plan_with_lists(start_list, goal_list, smoothing_enabled)

    def plan_with_lists(self, start_list: List, goal_list: List, smoothing_enabled: bool):
        try:
            path = self.planner.planPath(start_list, goal_list, self.config)
            print(len(path), "Size original path")
            graph = self.planner.graph
        except Exception as e:
            print(f"Error while planning with {self.name}: ")
            print(e)
            return None, self.planner.graph

        if smoothing_enabled:
            smoother = IPSmoothing(self.planner.graph, path, self.collision_checker)
            path, graph = smoother.smooth_solution(self.config["smoothing_max_iterations"],
                                                   self.config["smoothing_max_k"])
            print(len(path), "Size after smoothing")

        if path is not None:
            path = self._convert_path_to_PathNode(path, graph)
        return path, graph

    def _convert_to_start_goal_lists(self, start: Tuple, goal: Tuple) -> Tuple[List, List]:
        start_list = [round(start[0]), round(start[1])]
        goal_list = [round(goal[0]), round(goal[1])]
        return [start_list], [goal_list]

    def _convert_room_to_IPCollisionChecker(self, room):
        box = cv2.boundingRect(room.mask)
        IPlimits = [[box[0], box[0] + box[2]], [box[1], box[1] + box[3]]]
        IPscene = {str(i): obstacle for i, obstacle in enumerate(room.env.scene)}
        return CollisionChecker(IPscene, IPlimits)

    def _convert_from_point_to_IPCollisionChecker(self, point, max_dist, scene):
        IPlimits = [[round(point[0] - max_dist), round(point[0] + max_dist)],
                    [round(point[1] - max_dist), round(point[1] + max_dist)]]
        IPscene = {str(i): obstacle for i, obstacle in enumerate(scene)}
        return CollisionChecker(IPscene, IPlimits)

    def _convert_path_to_PathNode(self, path, graph) -> List:
        pos = nx.get_node_attributes(graph, 'pos')
        return [PathNode(Position(pos[node][0], pos[node][1], 0.0)) for node in path]


if __name__ == "__main__":
    import cv2
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    from semantic_hierarchical_graph.graph import SHGraph
    from semantic_hierarchical_graph.floor import Floor
    import semantic_hierarchical_graph.utils as util
    from semantic_hierarchical_graph.types.position import Position
    from semantic_hierarchical_graph.planners.astar_planner import AStarPlanner
    from semantic_hierarchical_graph.planners.rrt_planner import RRTPlanner

    G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    # room = floor._get_child("room_2")
    room = floor._get_child("room_11")
    # room = floor._get_child("room_14")

    # planner = RRTPlanner(room)
    planner = AStarPlanner(room)
    ts = time.time()
    path, vis_graph = planner.plan((480, 250), (75, 260), True)  # type: ignore
    # path, _ = planner.plan((555, 211), (81, 358), True)
    print("Time", time.time() - ts)
    path_list = util.map_names_to_nodes(path)

    print(path_list)
    print("Path length", len(path_list))
    path: List = path
    for i in range(len(path) - 1):
        pt1 = np.round(path[i].pos.xy).astype("int32")
        pt2 = np.round(path[i + 1].pos.xy).astype("int32")
        cv2.line(room.mask, pt1, pt2, (128), 1, cv2.LINE_4)
    plt.imshow(room.mask)
    plt.show()
