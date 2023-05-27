from collections import deque
from typing import Dict, List, Optional, Tuple
from shapely import LineString, Point

from semantic_hierarchical_graph.planners.planner_interface import PlannerInterface
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError, SHGPlannerError, SHGValueError
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
        self.tmp_node_added = []
        self.tmp_edge_added = []

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
            self.tmp_node_added = []
            self.tmp_edge_added = []

        return path, vis_graph, 0.0  # no smoothing time

    def _check_for_direct_connection(self, start_pos: Position, goal_pos: Position) -> Tuple:
        connection = self.room.env.get_valid_connection(Point(start_pos.xy), Point(goal_pos.xy))
        if connection is not None:
            print("Start point is closer to goal than to roadmap")
            path = [self.room._get_child(start_pos.to_name()),
                    self.room._get_child(goal_pos.to_name())]
            return path, start_pos.distance(goal_pos)

        return None, 0

    def _add_path_to_roadmap(self, node_name, node_pos, type, temporary=True):
        if self.room.env._in_collision(Point(node_pos.xy)):
            raise SHGPlannerError(
                f"Point {node_pos} is not in the drivable area (boundary + safety margin) of the room")

        connections, closest_path = roadmap_creation.connect_point_to_path(node_pos.xy, self.room.env, self.room.params)

        if len(connections) == 0:
            raise SHGPlannerError(f"No connection from point {node_pos.xy} to roadmap found")

        nodes = deque(maxlen=2)
        nodes.append(self._add_tmp_node(self.room,
                                        Position.from_iter(connections[0].coords[0]),
                                        type, temporary))
        distance = 0.0
        for connection in connections:
            pos = Position.from_iter(connection.coords[1])
            nodes.append(self._add_tmp_node(self.room, pos, type, temporary))
            d = nodes[0].pos.distance(nodes[1].pos)
            distance += d
            self._add_tmp_connection(self.room, nodes[0], nodes[1], d, temporary)

        if len(closest_path.coords) != 2:
            raise SHGGeometryError("Path has not 2 points as expected")

        path_1_pos = Position.from_iter(closest_path.coords[0])
        path_2_pos = Position.from_iter(closest_path.coords[1])
        path_1_node = self.room._get_child(path_1_pos.to_name())
        path_2_node = self.room._get_child(path_2_pos.to_name())

        # Remove existing edge to add junction in between
        if self.room.child_graph.has_edge(path_1_node, path_2_node):
            edge_data = self.room.child_graph.get_edge_data(path_1_node, path_2_node)
            # print(f"Removing edge {edge_data} from roadmap")
            self.room.child_graph.remove_edge(path_1_node, path_2_node)
            self.room.env.path.remove(closest_path)
            if temporary:
                self.tmp_edge_removed.append(
                    (path_1_node.unique_name, path_2_node.unique_name, edge_data, closest_path))

        self._add_tmp_connection(self.room, path_1_node, nodes[1], path_1_pos.distance(nodes[1].pos), temporary)
        self._add_tmp_connection(self.room, nodes[1], path_2_node, path_2_pos.distance(nodes[1].pos), temporary)

        return distance

    def _add_tmp_node(self, room_node: Room, node_pos, type, temporary):
        try:
            node = room_node._get_child(node_pos.to_name())
        except SHGValueError:
            node = room_node.add_child_by_name(node_pos.to_name(), node_pos, True, type=type)
            if temporary:
                self.tmp_node_added.append(node_pos.to_name())
            # print(f"Added node {node.unique_name} to roadmap")
        return node

    def _add_tmp_connection(self, room_node: Room, node_1, node_2, distance, temporary):
        if not room_node.child_graph.has_edge(node_1, node_2):
            room_node.add_connection_by_nodes(node_1, node_2, distance)
            path = LineString([node_1.pos.xy, node_2.pos.xy])
            if temporary:
                self.tmp_edge_added.append((node_1.unique_name, node_2.unique_name))

            if path not in room_node.env.path:
                room_node.env.add_path(path)
                if temporary:
                    self.tmp_path_added.append(path)
                # print(f"Added path {path} to roadmap")
            # print(f"Added edge {node_1.unique_name, node_2.unique_name} to roadmap")

    def _revert_tmp_graph(self):
        # print("========CLEANUP========")
        # The order of reverting is relevant!

        for edge in list(self.tmp_edge_added):
            if edge[0] in self.room.get_childs("name") and edge[1] in self.room.get_childs("name"):
                node_1 = self.room._get_child(edge[0])
                node_2 = self.room._get_child(edge[1])
                if self.room.child_graph.has_edge(node_1, node_2):
                    self.room.child_graph.remove_edge(node_1, node_2)
                    # print(f"Removed edge {edge[0], edge[1]} from roadmap")

        for node_name in self.tmp_node_added:
            # print(f"Removing node {node_name} from roadmap")
            if node_name in self.room.get_childs("name"):
                node = self.room._get_child(node_name)
                self.room.child_graph.remove_node(node)
                # print(f"Removed node {node_name} from roadmap")

        for path in list(self.tmp_path_added):
            if path in self.room.env.path:
                self.room.env.path.remove(path)
                # print(f"Removed path {path} from env")

        for edge in list(self.tmp_edge_removed):
            if edge[0] in self.room.get_childs("name") and edge[1] in self.room.get_childs("name"):
                node_1 = self.room._get_child(edge[0])
                node_2 = self.room._get_child(edge[1])
                self.room.child_graph.add_edge(node_1, node_2, **edge[2])
                self.room.env.add_path(edge[3])
                # print(f"Added path {edge[3]} to env")
                # print(f"Added edge {edge[2]} to roadmap")


if __name__ == "__main__":
    # G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    # floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    # G.add_child_by_node(floor)
    # print(G.get_childs("name"))

    # floor.create_rooms()
    # floor.create_bridges()

    # G.save_graph("data/tmp/graph.pickle")
    G = SHGraph.load_graph("data/tmp/graph.pickle")
    floor = G._get_child("ryu")

    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    room_14 = floor._get_child("room_14")
    planner = ILIRPlanner(room_11)
    segmentation.show_imgs(room_11.mask)
    path, vis_graph, distance = planner.plan((480, 250), (75, 260))
    # path = planner.plan((555, 211), (81, 358))
    path_list = util.map_names_to_nodes(path)

    print(path_list)
    vis.draw_child_graph_3d(room_11, path, vis_graph)
