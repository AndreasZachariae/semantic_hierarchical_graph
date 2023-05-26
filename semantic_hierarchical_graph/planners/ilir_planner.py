from collections import deque
from typing import Dict, List, Optional, Tuple
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
            config["smoothing_algorithm"] = "bechtold_glavina"
            config["smoothing_max_iterations"] = 100
            config["smoothing_max_k"] = 50

        super().__init__(room, config)

        self.name = "ILIR"
        self.graph = self.room.child_graph
        self.planner = self

        self.tmp_edge_removed = []
        self.tmp_path_added = []

    def plan_with_lists(self, start_list: List, goal_list: List, smoothing_enabled: bool = False):
        return self.plan(start_list[0], goal_list[0], False)

    def plan(self, start: Tuple, goal: Tuple, smoothing_enabled: bool = False):
        path = None
        try:
            start_pos = Position.from_iter(start)
            goal_pos = Position.from_iter(goal)
            distance_to_roadmap = 0.0
            if start_pos.xy in self.room.bridge_points_not_connected or goal_pos.xy in self.room.bridge_points_not_connected:
                raise SHGPlannerError(f"This bridge point {start_pos} can not be connected to the roadmap")
            if start_pos.to_name() not in self.room.get_childs("name"):
                distance_to_roadmap += self._add_path_to_roadmap(start_pos.to_name(), start_pos, type="start")
            if goal_pos.to_name() not in self.room.get_childs("name"):
                distance_to_roadmap += self._add_path_to_roadmap(goal_pos.to_name(), goal_pos, type="goal")

            # if distance_to_roadmap > start_pos.distance(goal_pos):
            #     path, distance = self._check_for_direct_connection(start_pos, goal_pos)

            if not path:
                path = self.room.plan_in_graph(start_pos.to_name(), goal_pos.to_name())
        except SHGPlannerError as e:
            print("Error while planning with ILIRPlanner: ")
            print(e)
            path = None
        finally:
            vis_graph = self.room.child_graph
            self._revert_tmp_graph()
            self.tmp_edge_removed = []
            self.tmp_path_added = []

        return path, vis_graph, 0.0  # no smoothing time

    def _check_for_direct_connection(self, start_pos: Position, goal_pos: Position) -> Tuple:
        connection = self.room.env.get_valid_connection(Point(start_pos.xy), Point(goal_pos.xy))
        if connection is not None:
            print("Start point is closer to goal than to roadmap")
            path = [self.room._get_child(start_pos.to_name()),
                    self.room._get_child(goal_pos.to_name())]
            return path, start_pos.distance(goal_pos)

        return None, 0

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
        distance = 0.0
        for connection in connections:
            pos = Position.from_iter(connection.coords[1])
            nodes.append(self.room.add_child_by_name(pos.to_name(), pos, True, type="aux_node"))
            d = nodes[0].pos.distance(nodes[1].pos)
            distance += d
            self.room.add_connection_by_nodes(nodes[0], nodes[1], d)

        if len(closest_path.coords) != 2:
            raise SHGGeometryError("Path has not 2 points as expected")

        path_1_pos = Position.from_iter(closest_path.coords[0])
        path_2_pos = Position.from_iter(closest_path.coords[1])
        path_1_node = self.room._get_child(path_1_pos.to_name())
        path_2_node = self.room._get_child(path_2_pos.to_name())
        if self.room.child_graph.has_edge(path_1_node, path_2_node):
            edge_data = self.room.child_graph.get_edge_data(path_1_node, path_2_node)
            self.room.child_graph.remove_edge(path_1_node, path_2_node)
            self.room.env.path.remove(closest_path)
            self.tmp_edge_removed.append((path_1_node, path_2_node, edge_data, closest_path))

        self.room.add_connection_by_nodes(path_1_node, nodes[1], path_1_pos.distance(nodes[1].pos))
        self.room.add_connection_by_nodes(nodes[1], path_2_node, path_2_pos.distance(nodes[1].pos))
        path1 = LineString([path_1_pos.xy, nodes[1].pos.xy])
        path2 = LineString([nodes[1].pos.xy, path_2_pos.xy])
        self.tmp_path_added.append(path1)
        self.tmp_path_added.append(path2)
        self.room.env.add_path(path1)
        self.room.env.add_path(path2)

        return distance

    def _revert_tmp_graph(self):
        to_remove = []
        for node in self.room.get_childs():
            if node.type in ["start", "goal", "aux_node"]:
                to_remove.append(node)
        for node in to_remove:
            self.room.child_graph.remove_node(node)

        for edge in list(self.tmp_edge_removed):
            if edge[0] in self.room.child_graph.nodes and edge[1] in self.room.child_graph.nodes:
                self.room.child_graph.add_edge(edge[0], edge[1], **edge[2])
                self.room.env.add_path(edge[3])
                # print(f"Added edge {edge[2]} to roadmap")
                self.tmp_edge_removed.remove(edge)

        for path in list(self.tmp_path_added):
            if path in self.room.env.path:
                self.room.env.path.remove(path)
                self.tmp_path_added.remove(path)

        # print(f"Removed {len(to_remove)} nodes for cleanup")


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
    segmentation.show_imgs(room_11.mask)
    path, vis_graph = planner.plan((480, 250), (75, 260))
    # path = planner.plan((555, 211), (81, 358))
    path_list = util.map_names_to_nodes(path)

    print(path_list)
    vis.draw_child_graph_3d(room_11, path, vis_graph)
