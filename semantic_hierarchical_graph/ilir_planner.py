from collections import deque
from typing import List, Tuple, Union

from shapely import Point
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.floor import Room, Floor
from semantic_hierarchical_graph import path_planning, segmentation, visualization as vis
from semantic_hierarchical_graph.node import SHNode
import semantic_hierarchical_graph.utils as util
from semantic_hierarchical_graph.types import Position


class ILIRPlanner():
    def __init__(self, room: Room):
        self.room = room
        self.original_graph = room.child_graph.copy()
        self.original_leaf_graph = room._get_root_node().leaf_graph.copy()
        self.tmp_graph = room.child_graph

    def plan(self, start: Union[Tuple, List, Position], goal: Union[Tuple, List, Position]):
        start_name, start_pos = self._convert_position_to_grid_node(start)
        goal_name, goal_pos = self._convert_position_to_grid_node(goal)
        if start_name not in self.room.get_childs("name"):
            self._add_path_to_roadmap(start_name, start_pos, type="start")
        if goal_name not in self.room.get_childs("name"):
            self._add_path_to_roadmap(goal_name, goal_pos, type="goal")

        path = self.room._plan(start_name, goal_name)

        # Reset graph
        self.room.child_graph = self.original_graph
        self.room._get_root_node().leaf_graph = self.original_leaf_graph

        return path

    def _convert_position_to_grid_node(self, pos: Union[Tuple, List, Position]) -> Tuple[str, Position]:
        # TODO: Convert meters from map frame to pixels in grid map
        if not isinstance(pos, Position):
            pos = Position.from_iter(pos)
        return pos.to_name(), pos

    def _add_path_to_roadmap(self, node_name, node_pos, type):
        if self.room.env._in_collision(Point(node_pos.xy)):
            raise Exception(f"Point {node_pos} is not in the drivable area (boundary + safety margin) of the room")

        connections, closest_path = path_planning.connect_point_to_path(node_pos.xy, self.room.env, self.room.params)

        if len(connections) == 0:
            raise Exception(f"No connection from point {node_pos} to roadmap found")

        nodes = deque(maxlen=2)
        prev_node = self.room.add_child_by_name(node_name,
                                                Position.from_iter(connections[0].coords[0]),
                                                True, type=type)
        for connection in connections:
            for point in connection.coords:
                pos = Position.from_iter(connection.coords[0])
                cur_node = self.room.add_child_by_name(pos.to_name(), pos, True, type="aux_node")
                self.room.add_connection_by_nodes(prev_node, cur_node,
                                                  prev_node.pos.distance(cur_node.pos))
                prev_node = cur_node

        if len(closest_path.coords) != 2:
            raise ValueError("Path has not 2 points as expected")

        path_1_pos = Position.from_iter(closest_path.coords[0])
        path_2_pos = Position.from_iter(closest_path.coords[1])
        path_1_node = self.room._get_child(path_1_pos.to_name())
        path_2_node = self.room._get_child(path_2_pos.to_name())
        self.room.child_graph.remove_edge(path_1_node, path_2_node)
        self.room.add_connection_by_nodes(path_1_node, prev_node, path_1_pos.distance(prev_node.pos))
        self.room.add_connection_by_nodes(prev_node, path_2_node, path_2_pos.distance(prev_node.pos))

        print(self.room.get_childs("name"))
        # vis.draw_child_graph_3d(self.room)


if __name__ == "__main__":
    G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    room_14 = floor._get_child("room_14")
    planner = ILIRPlanner(room_11)
    # TODO: test for goal point not on graph
    # segmentation.show_imgs(room_11.mask)
    path = planner.plan((480, 250), (75, 260))
    # path = planner.plan((555, 211), (81, 358))
    path_list = util._map_names_to_nodes(path)

    print(path_list)
    vis.draw_child_graph_3d(room_11, path)  # type: ignore
