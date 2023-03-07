from collections import deque
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from shapely import LineString, Point
from semantic_hierarchical_graph.planners.planner_interface import PlannerInterface
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError, SHGPlannerError
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.floor import Room, Floor
from semantic_hierarchical_graph import roadmap_creation, segmentation, visualization as vis
import semantic_hierarchical_graph.utils as util
from semantic_hierarchical_graph.types.position import Position


class ILIRPlanner(PlannerInterface):
    def __init__(self, room: Room, config: Optional[Dict] = None):
        if config is None:
            config = dict()
            config["smoothing_iterations"] = 50
            config["smoothing_max_k"] = 20
            config["smoothing_epsilon"] = 0.5
            config["smoothing_variance_window"] = 10
            config["smoothing_min_variance"] = 0.0

        super().__init__(room, config)

        self.name = "ILIR"
        self.graph = self.room.child_graph
        self.planner = self

    def _copy_graph(self):
        # TODO: This is a hack to make sure that the graph is not modified by the planner
        #       This takes very long and slows down the planning
        self.original_path = deepcopy(self.room.env.path)
        self.original_graph = deepcopy(self.room.child_graph)
        self.original_leaf_graph = deepcopy(self.room._get_root_node().leaf_graph)

    def _reset_graph(self):
        self.room.env.path = self.original_path
        self.room.child_graph = self.original_graph
        self.room._get_root_node().leaf_graph = self.original_leaf_graph

    def plan_with_lists(self, start_list: List, goal_list: List, smoothing_enabled: bool = False):
        return self.plan(start_list[0], goal_list[0], False)

    def plan(self, start: Tuple, goal: Tuple, smoothing_enabled: bool = False):
        self._copy_graph()
        try:
            start_pos = Position.from_iter(start)
            goal_pos = Position.from_iter(goal)
            if start_pos.xy in self.room.bridge_points_not_connected or goal_pos.xy in self.room.bridge_points_not_connected:
                raise SHGPlannerError(f"This bridge point {start_pos} can not be connected to the roadmap")
            if start_pos.to_name() not in self.room.get_childs("name"):
                self._add_path_to_roadmap(start_pos.to_name(), start_pos, type="start")
            if goal_pos.to_name() not in self.room.get_childs("name"):
                self._add_path_to_roadmap(goal_pos.to_name(), goal_pos, type="goal")

            path = self.room._plan(start_pos.to_name(), goal_pos.to_name())
        except SHGPlannerError as e:
            print("Error while planning with ILIRPlanner: ")
            print(e)
            path = None
        finally:
            vis_graph = self.room.child_graph
            self._reset_graph()

        return path, vis_graph

    def _add_path_to_roadmap(self, node_name, node_pos, type):
        if self.room.env._in_collision(Point(node_pos.xy)):
            raise SHGPlannerError(
                f"Point {node_pos} is not in the drivable area (boundary + safety margin) of the room")

        connections, closest_path = roadmap_creation.connect_point_to_path(node_pos.xy, self.room.env, self.room.params)

        if len(connections) == 0:
            raise SHGPlannerError(f"No connection from point {node_pos.xy} to roadmap found")

        nodes = deque(maxlen=2)
        nodes.append(self.room.add_child_by_name(node_name,
                                                 Position.from_iter(connections[0].coords[0]),
                                                 True, type=type))
        for connection in connections:
            pos = Position.from_iter(connection.coords[1])
            nodes.append(self.room.add_child_by_name(pos.to_name(), pos, True, type="aux_node"))
            self.room.add_connection_by_nodes(nodes[0], nodes[1],
                                              nodes[0].pos.distance(nodes[1].pos))

        if len(closest_path.coords) != 2:
            raise SHGGeometryError("Path has not 2 points as expected")

        path_1_pos = Position.from_iter(closest_path.coords[0])
        path_2_pos = Position.from_iter(closest_path.coords[1])
        path_1_node = self.room._get_child(path_1_pos.to_name())
        path_2_node = self.room._get_child(path_2_pos.to_name())
        # TODO: In some cases the edge is not in the graph. Could be a logic error. Need to fix!
        self.room.child_graph.remove_edge(path_1_node, path_2_node)
        self.room.env.path.remove(closest_path)
        self.room.add_connection_by_nodes(path_1_node, nodes[1], path_1_pos.distance(nodes[1].pos))
        self.room.add_connection_by_nodes(nodes[1], path_2_node, path_2_pos.distance(nodes[1].pos))
        self.room.env.add_path(LineString([path_1_pos.xy, nodes[1].pos.xy]))
        self.room.env.add_path(LineString([nodes[1].pos.xy, path_2_pos.xy]))


if __name__ == "__main__":
    G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu_clean.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    room_14 = floor._get_child("room_14")
    planner = ILIRPlanner(room_11)
    # segmentation.show_imgs(room_11.mask)
    path, vis_graph = planner.plan((480, 250), (75, 260))
    # path = planner.plan((555, 211), (81, 358))
    path_list = util._map_names_to_nodes(path)

    print(path_list)
    vis.draw_child_graph_3d(room_11, path, vis_graph)
