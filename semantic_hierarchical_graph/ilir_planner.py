from typing import List, Tuple
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.floor import Room, Floor
from semantic_hierarchical_graph import visualization as vis
import semantic_hierarchical_graph.utils as util


Position = Tuple[float, float, float]

class ILIRPlanner():
    def __init__(self, room: Room):
        self.room = room

    def plan(self, start: Position, goal: Position):
        grid_start = self._convert_position_to_grid(start)
        grid_goal = self._convert_position_to_grid(goal)
        start_path, goal_path = [], []
        if str(grid_start) not in self.room.get_childs("name"):
            start_path = self._path_to_roadmap(grid_start)
        if str(grid_goal) not in self.room.get_childs("name"):
            goal_path = self._path_to_roadmap(grid_goal)

        path = self.room._plan(str(grid_start), str(grid_goal))
        path.extend(goal_path)
        start_path.extend(path)

        return start_path

    def _convert_position_to_grid(self, pos: Position) -> Tuple[int, int]:
        # TODO: Convert meters from map frame to pixels in grid map
        return (round(pos[0]), round(pos[1]))

    def _path_to_roadmap(self, node) -> List:
        return []

if __name__ == "__main__":
    G = SHGraph(root_name="Benchmark", root_pos=(0, 0, 0))
    floor = Floor("ryu", G, (0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    room_14 = floor._get_child("room_14")
    planner = ILIRPlanner(room_11)
    path = planner.plan((555, 211), (81, 358))
    path_list = util._map_names_to_nodes(path)

    print(path_list)
    vis.draw_child_graph_3d(room_11, path)
