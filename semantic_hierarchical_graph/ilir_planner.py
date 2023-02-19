from typing import List, Tuple

from shapely import Point
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.floor import Room, Floor
from semantic_hierarchical_graph import path_planning, segmentation, visualization as vis
from semantic_hierarchical_graph.node import SHNode
import semantic_hierarchical_graph.utils as util


Position = Tuple[float, float, float]

class ILIRPlanner():
    def __init__(self, room: Room):
        self.room = room
        self.original_graph = room.child_graph.copy()
        self.original_leaf_graph = room._get_root_node().leaf_graph.copy()
        self.tmp_graph = room.child_graph

    def plan(self, start: Position, goal: Position):
        start_name, start_pos = self._convert_position_to_grid_node(start)
        goal_name, goal_pos = self._convert_position_to_grid_node(goal)
        if start_name not in self.room.get_childs("name"):
            self._add_path_to_roadmap(start_name, start_pos, type = "start")
        if goal_name not in self.room.get_childs("name"):
            self._add_path_to_roadmap(goal_name, goal_pos, type = "goal")

        path = self.room._plan(start_name, goal_name)

        # Reset graph
        self.room.child_graph = self.original_graph
        self.room._get_root_node().leaf_graph = self.original_leaf_graph

        return path

    def _convert_position_to_grid_node(self, pos: Position) -> Tuple[str, Position]:
        # TODO: Convert meters from map frame to pixels in grid map
        return str((round(pos[0]), round(pos[1]))), pos

    def _add_path_to_roadmap(self, node_name, node_pos, type):
        if self.room.env._in_collision(Point(node_pos[:2])):
            raise Exception(f"Point {node_pos} is not in the drivable area (boundary + safety margin) of the room")
        
        connections, closest_path = path_planning.connect_point_to_path(node_pos[:2], self.room.env, self.room.params)

        if len(connections) == 0:
            raise Exception(f"No connection from point {node_pos} to roadmap found")
        
        prev_node = self.room.add_child_by_name(node_name, node_pos, True, type=type)
        for connection in connections:
            for point in connection.coords:
                p = str((round(point[0]), round(point[1])))
                cur_node = self.room.add_child_by_name(p, point + (0.0,), True, type="aux_node")
                self.room.add_connection_by_nodes(prev_node, cur_node, 
                                                  util.get_euclidean_distance(prev_node.pos, cur_node.pos))
                prev_node = cur_node

        if len(closest_path.coords) != 2:
            raise ValueError("Path has not 2 points as expected")
        path_1_node = self.room._get_child(str((round(closest_path.coords[0][0]), round(closest_path.coords[0][1]))))
        path_2_node = self.room._get_child(str((round(closest_path.coords[1][0]), round(closest_path.coords[1][1]))))
        self.room.child_graph.remove_edge(path_1_node, path_2_node)
        self.room.add_connection_by_nodes(path_1_node, prev_node, 
                                          util.get_euclidean_distance(closest_path.coords[0]+ (0.0,), prev_node.pos))
        self.room.add_connection_by_nodes(prev_node, path_2_node, 
                                          util.get_euclidean_distance(closest_path.coords[1]+ (0.0,), prev_node.pos))

        # vis.draw_child_graph_3d(self.room)


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
    # TODO: test for goal point not on graph
    path = planner.plan((480, 250, 0), (81, 358, 0))
    # path = planner.plan((555, 211, 0), (81, 358, 0))
    path_list = util._map_names_to_nodes(path)

    print(path_list)
    vis.draw_child_graph_3d(room_11, path)  # type: ignore
