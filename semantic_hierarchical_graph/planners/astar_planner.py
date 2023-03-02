from typing import Optional, Tuple

from path_planner_suite.IPAStar import AStar
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
        else:
            self.config = config

        self.benchmark = pc.convert_room_to_IPBenchmark(room)
        self.planner = AStar(self.benchmark.collisionChecker)

    def plan(self, start: Tuple, goal: Tuple):
        start_list, goal_list = pc.convert_to_start_goal_lists(start, goal)
        try:
            path = self.planner.planPath(start_list, goal_list, self.config)
        except Exception as e:
            print("Error while planning with AStarPlanner: ")
            print(e)
            path = None

        if path is not None:
            path = pc.convert_path_to_PathNode(path, self.planner.graph)
        return path, self.planner.graph

    def plan_in_map_frame(self, start: Tuple, goal: Tuple):
        start_pos, goal_pos = pc.convert_map_frame_to_grid(start, goal, self.room.params["grid_size"])
        self.plan(start_pos, goal_pos)


if __name__ == "__main__":
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
    # segmentation.show_imgs(room_11.mask)
    path, vis_graph = planner.plan((480, 250), (75, 260))
    # path = planner.plan((555, 211), (81, 358))
    path_list = util._map_names_to_nodes(path)

    print(path_list)
    # vis.draw_child_graph_3d(room_11, path, vis_graph)
