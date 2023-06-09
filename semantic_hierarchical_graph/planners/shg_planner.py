from collections import deque
import os
from typing import Dict, List, Tuple
import numpy as np
from shapely import LineString, Point
import math
import networkx as nx

from semantic_hierarchical_graph import roadmap_creation, segmentation, visualization as vis
from semantic_hierarchical_graph.floor import Floor, Room
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.path import SHPath
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError, SHGPlannerError, SHGValueError
from semantic_hierarchical_graph.types.parameter import Parameter
from semantic_hierarchical_graph.types.position import Position
from semantic_hierarchical_graph.types.vector import Vector


class SHGPlanner():
    def __init__(self, graph_path: str, graph_name: str = "graph.pickle", force_create: bool = False):
        self.graph_path = graph_path
        self.graph: SHGraph = self._init_graph(graph_name, force_create)
        self.path: Dict = {}
        self.distance: float = 0.0
        self.tmp_edge_removed = []
        self.tmp_edge_added = []
        self.tmp_path_added = []
        self.tmp_node_added = []

        self.name = "SHG"
        self.config = self.graph.params

        self.current_map_origin: Tuple[float, float]
        self.current_map_resolution: float
        self.current_map_shape: Tuple[int, int]
        self.current_floor_name: str

    def _init_graph(self, graph_name: str, force_create: bool) -> SHGraph:
        if not force_create and os.path.isfile(self.graph_path + "/" + graph_name):
            return self._load_graph(graph_name)

        return self._create_graph()

    def _load_graph(self, graph_name: str) -> SHGraph:
        G = SHGraph.load_graph(self.graph_path + "/" + graph_name)
        G.params = Parameter(self.graph_path + "/graph.yaml", is_map=False).params
        self._load_connections(G)
        return G

    def _create_graph(self) -> SHGraph:
        params = Parameter(self.graph_path + "/graph.yaml", is_map=False).params
        G = SHGraph(root_name=self.graph_path.split("/")[-1], root_pos=Position(0, 0, 0), params=params)

        for floor_nr, floor_config in enumerate(params["maps"]):
            floor_path = os.path.join(self.graph_path, floor_config["yaml_path"])
            floor_name = floor_config["hierarchy"][-1]
            floor_map = floor_path.replace(".yaml", ".pgm")
            print("Creating floor: " + floor_name)
            floor = Floor(floor_name, G, Position(0, 0, floor_nr), floor_map, floor_path)
            G.add_child_by_node(floor)

            floor.create_rooms()
            floor.create_bridges()

        self._load_connections(G)

        G.save_graph(self.graph_path + "/graph.pickle")

        return G

    def _load_connections(self, graph: SHGraph):
        for connection in graph.params["connections"]:

            self._check_connection_nodes_in_roadmap(graph, connection, 0)
            self._check_connection_nodes_in_roadmap(graph, connection, 1)

            graph.add_connection_recursive(connection["hierarchy"][0], connection["hierarchy"][1],
                                           distance=connection["cost"], name=connection["name"])

            print(connection["hierarchy"], "added to graph")

    def _check_connection_nodes_in_roadmap(self, graph: SHGraph, connection: Dict, id: int):
        pos = Position.from_iter(connection["hierarchy"][id][-1])
        connection["hierarchy"][id][-1] = pos.to_name()

        try:
            graph.get_child_by_hierarchy(connection["hierarchy"][id])
        except SHGValueError:
            room_node: Room = graph.get_child_by_hierarchy(connection["hierarchy"][id][:-1])  # type: ignore
            self._add_path_to_roadmap(room_node, pos.to_name(), pos, connection["name"], temporary=False)

        connection_node = graph.get_child_by_hierarchy(connection["hierarchy"][id])
        connection_node.data_dict["call_button_angle"] = connection["call_button"]["angle"]
        connection_node.data_dict["call_button_marker_id_up"] = connection["call_button"]["marker_id_up"]
        connection_node.data_dict["call_button_marker_id_down"] = connection["call_button"]["marker_id_down"]
        connection_node.data_dict["waiting_pose_start_orientation"] = connection["waiting_pose_start"]["orientation"]
        connection_node.data_dict["waiting_pose_start_position"] = connection["waiting_pose_start"]["position"]
        connection_node.data_dict["waiting_pose_goal_orientation"] = connection["waiting_pose_goal"]["orientation"]
        connection_node.data_dict["waiting_pose_goal_position"] = connection["waiting_pose_goal"]["position"]
        connection_node.data_dict["panel_point_start"] = connection["panel_point_start"]
        connection_node.data_dict["panel_point_goal"] = connection["panel_point_goal"]

    def update_floor(self, origin: Tuple[float, float], resolution: float, map_shape: Tuple[int, int], current_floor: str):
        self.current_map_origin = origin
        self.current_map_resolution = resolution
        self.current_map_shape = map_shape
        self.current_floor_name = current_floor

    def add_location(self, hierarchy, location):
        # TODO: add location to graph and save graph
        pass

    def _get_hierarchy(self, pos: Position, floor: str) -> List:
        if not isinstance(pos, Position):
            pos = Position.from_iter(pos)

        floor_node: Floor = self.graph._get_child(floor)
        room_id = int(floor_node.watershed[int(pos.y), int(pos.x)])  # type: ignore

        if room_id == 0 or room_id == 1:
            raise SHGPlannerError("Position is not in a valid room")

        if room_id == -1:
            try:
                rooms = [adj_rooms for adj_rooms, edges in floor_node.bridge_edges.items()
                         for edge in edges if pos.xy in edge]
                # room_id = np.random.choice(rooms[0])
                room_id = rooms[0][0]
            except:
                raise SHGPlannerError("Position is not in a valid room")

        room_node: Room = self.graph.get_child_by_hierarchy([floor, "room_" + str(room_id)])  # type: ignore

        return [floor_node, room_node, pos]

    def plan_in_map_frame(self, start_pose: Tuple, start_floor: str, goal_pose: Tuple, goal_floor: str, interpolation_resolution: float) -> Tuple[List, float]:
        """
        Plan a path in the map frame
        :param start_pose: start pose as ROS2 PoseStamped
        :param goal_pose: goal pose as ROS2 PoseStamped
        :return: path as list of coordiantes in map frame
        """
        start = Position.from_map_frame(start_pose, self.current_map_origin,
                                        self.current_map_resolution, self.current_map_shape)
        goal = Position.from_map_frame(goal_pose, self.current_map_origin,
                                       self.current_map_resolution, self.current_map_shape)

        path, distance = self._plan([start_floor, None, start], [goal_floor, None, goal])

        path_list = self.get_path_on_floor([self.current_floor_name], "position",
                                           interpolation_resolution / self.current_map_resolution)

        path_list = [p.to_map_frame(self.current_map_origin, self.current_map_resolution,
                                    self.current_map_shape) for p in path_list]

        return path_list, distance

    def plan_on_floor(self, floor_name: str, start_pose: Tuple, goal_pose: Tuple, smoothing: bool = True) -> Tuple[List, nx.Graph, float]:
        path, distance = self._plan([floor_name, None, start_pose], [floor_name, None, goal_pose])
        return self.get_path_on_floor([floor_name], "node"), self.graph._get_child(floor_name).child_graph, 0.0

    def _plan(self, start: List, goal: List) -> Tuple[Dict, float]:
        """Expects hierarchy in type [str, str, Position/Tuple]"""
        self.path = {}
        try:
            start = self._get_hierarchy(start[2], start[0])
            goal = self._get_hierarchy(goal[2], goal[0])
        except SHGPlannerError as e:
            print(e)
            return {}, 0.0

        try:
            print("Plan from " + str(start[2].to_name()) + " to " + str(goal[2].to_name()))

            distance_to_roadmap = 0.0
            if start[2].to_name() not in start[1].get_childs("name"):
                distance_to_roadmap += self._add_path_to_roadmap(start[1],
                                                                 start[2].to_name(), start[2], type="start")
            if goal[2].to_name() not in goal[1].get_childs("name"):
                distance_to_roadmap += self._add_path_to_roadmap(
                    goal[1], goal[2].to_name(), goal[2], type="goal")

            if start[1] == goal[1]:
                if distance_to_roadmap > start[2].distance(goal[2]):
                    self.path, self.distance = self._check_for_direct_connection(
                        start[1], start[2], goal[2])

            if not self.path:
                self.path, self.distance = self.graph.plan_recursive(
                    [start[0].unique_name, start[1].unique_name, start[2].to_name()],
                    [goal[0].unique_name, goal[1].unique_name, goal[2].to_name()])

            # vis.draw_child_graph(start_room, self.path)
        except SHGPlannerError as e:
            print("Error while planning with SHGPlanner: ")
            print(e)
        finally:
            self._revert_tmp_graph(start[1])
            self._revert_tmp_graph(goal[1])
            self.tmp_edge_removed = []
            self.tmp_path_added = []
            self.tmp_node_added = []
            self.tmp_edge_added = []

        # SHPath.save_path(self.path, self.graph_path + "/path.json")

        return self.path, self.distance

    def _check_for_direct_connection(self, start_room: Room, start_pos: Position, goal_pos: Position) -> Tuple:
        connection = start_room.env.get_valid_connection(Point(start_pos.xy), Point(goal_pos.xy))
        if connection is not None:
            print("Start point is closer to goal than to roadmap")
            path = {start_room.parent_node: {start_room:
                                             {start_room._get_child(start_pos.to_name()): {},
                                                 start_room._get_child(goal_pos.to_name()): {}}}}
            return path, start_pos.distance(goal_pos)

        return {}, 0

    def _add_path_to_roadmap(self, room_node: Room, node_name, node_pos, type, temporary=True) -> float:
        if room_node.env._in_collision(Point(node_pos.xy)):
            raise SHGPlannerError(
                f"Point {node_pos.xy} is not in the drivable area (boundary + safety margin) of the room")

        connections, closest_path = roadmap_creation.connect_point_to_path(node_pos.xy, room_node.env, room_node.params)

        if len(connections) == 0 or closest_path is None:
            raise SHGPlannerError(f"No connection from point {node_pos.xy} to roadmap found")

        nodes = deque(maxlen=2)
        nodes.append(self._add_tmp_node(room_node,
                                        Position.from_iter(connections[0].coords[0]),
                                        type, temporary))
        distance = 0.0
        for connection in connections:
            pos = Position.from_iter(connection.coords[1])
            nodes.append(self._add_tmp_node(room_node, pos, type, temporary))
            d = nodes[0].pos.distance(nodes[1].pos)
            distance += d
            self._add_tmp_connection(room_node, nodes[0], nodes[1], d, temporary)

        if len(closest_path.coords) != 2:
            raise SHGGeometryError("Path has not 2 points as expected")

        path_1_pos = Position.from_iter(closest_path.coords[0])
        path_2_pos = Position.from_iter(closest_path.coords[1])
        path_1_node = room_node._get_child(path_1_pos.to_name())
        path_2_node = room_node._get_child(path_2_pos.to_name())

        # Remove existing edge to add junction in between
        if room_node.child_graph.has_edge(path_1_node, path_2_node):
            edge_data = room_node.child_graph.get_edge_data(path_1_node, path_2_node)
            # print(f"Removing edge {edge_data} from roadmap")
            room_node.child_graph.remove_edge(path_1_node, path_2_node)
            room_node.env.path.remove(closest_path)
            if temporary:
                self.tmp_edge_removed.append(
                    (path_1_node.unique_name, path_2_node.unique_name, edge_data, closest_path))

        self._add_tmp_connection(room_node, path_1_node, nodes[1], path_1_pos.distance(nodes[1].pos), temporary)
        self._add_tmp_connection(room_node, nodes[1], path_2_node, path_2_pos.distance(nodes[1].pos), temporary)

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

    def _revert_tmp_graph(self, room_node):
        # print("========CLEANUP========")
        # The order of reverting is relevant!

        for edge in list(self.tmp_edge_added):
            if edge[0] in room_node.get_childs("name") and edge[1] in room_node.get_childs("name"):
                node_1 = room_node._get_child(edge[0])
                node_2 = room_node._get_child(edge[1])
                if room_node.child_graph.has_edge(node_1, node_2):
                    room_node.child_graph.remove_edge(node_1, node_2)
                    # print(f"Removed edge {edge[0], edge[1]} from roadmap")

        for node_name in self.tmp_node_added:
            # print(f"Removing node {node_name} from roadmap")
            if node_name in room_node.get_childs("name"):
                node = room_node._get_child(node_name)
                room_node.child_graph.remove_node(node)
                # print(f"Removed node {node_name} from roadmap")

        for path in list(self.tmp_path_added):
            if path in room_node.env.path:
                room_node.env.path.remove(path)
                # print(f"Removed path {path} from env")

        for edge in list(self.tmp_edge_removed):
            if edge[0] in room_node.get_childs("name") and edge[1] in room_node.get_childs("name"):
                node_1 = room_node._get_child(edge[0])
                node_2 = room_node._get_child(edge[1])
                room_node.child_graph.add_edge(node_1, node_2, **edge[2])
                room_node.env.add_path(edge[3])
                # print(f"Added path {edge[3]} to env")
                # print(f"Added edge {edge[2]} to roadmap")

    def _get_path_on_floor(self, hierarchy_to_floor, key, path) -> List:
        leaf_path = []
        for node, dict in path.items():
            if len(hierarchy_to_floor) > 0 and node.unique_name != hierarchy_to_floor[0]:
                continue
            if node.is_leaf and not "bridge" in node.unique_name:
                if key == "name":
                    leaf_path.append(node.unique_name)
                elif key == "position":
                    leaf_path.append(node.pos)
                elif key == "node":
                    leaf_path.append(node)
            else:
                leaf_path.extend(self._get_path_on_floor(hierarchy_to_floor[1:], key, dict))
        return leaf_path

    def get_path_on_floor(self, hierarchy_to_floor, key, interpolation_resolution=None) -> List:
        """Returns the path on a specific floor as a list of nodes, positions or names."""
        if not self.path:
            print("No path found yet. Call plan() first.")
            return []

        path_list: List[Position] = self._get_path_on_floor(hierarchy_to_floor, key, self.path)
        path_list = list(dict.fromkeys(path_list))  # remove duplicates

        if key == "position":
            for i in range(len(path_list)-1):
                if path_list[i].rz is None:
                    angle = Vector.from_two_points(path_list[i].xy, path_list[i+1].xy).angle_to_grid()
                    path_list[i].rz = angle
                    # print(path_list[i].xy, path_list[i+1].xy, angle)

            # last node on floor has to get call button orientation defined in graph.yaml
            hierarchy = self._get_hierarchy(path_list[-1], hierarchy_to_floor[0])
            hierarchy_to_floor.append("room_" + str(hierarchy[1].id))
            hierarchy_to_floor.append(path_list[-1].to_name())
            try:
                call_button_angle = self.graph.get_child_by_hierarchy(
                    hierarchy_to_floor).data_dict.get("call_button_angle")
                if call_button_angle is not None:
                    path_list[-1].rz = call_button_angle
            except:
                pass

        if interpolation_resolution is not None:
            path_list = self._interpolate_path(path_list, interpolation_resolution)

        return path_list

    def _interpolate_path(self, path_list, resolution):
        interpolated_path = []
        for i in range(len(path_list) - 1):
            interpolated_path.append(path_list[i])
            d = path_list[i].distance(path_list[i+1])
            n = math.ceil(d / resolution)
            interpolated_xy = np.linspace(path_list[i].xy, path_list[i+1].xy, n, endpoint=False)
            for j in range(n):
                interpolated_path.append(Position(interpolated_xy[j][0], interpolated_xy[j][1], 0, path_list[i].rz))

        interpolated_path.append(path_list[-1])
        return interpolated_path

    def draw_path(self, save=False, name="path.png"):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        num_floors = len(self.graph.get_childs())
        fig = plt.figure(figsize=(15, 5*num_floors))

        for n, floor in enumerate(self.graph.get_childs()):
            floor_img = segmentation.draw(floor.watershed, floor.all_bridge_nodes, (22))
            for room in floor.get_childs():
                try:
                    [cv2.polylines(floor_img, [line.coords._coords.astype("int32")], False,  (0), 2)
                     for line in room.env.path]
                except AttributeError:
                    continue

            path_list = self._get_path_on_floor(floor.hierarchy, "node", self.path)
            for i in range(len(path_list) - 1):
                pt1 = np.round(path_list[i].pos.xy).astype("int32")  # type: ignore
                pt2 = np.round(path_list[i + 1].pos.xy).astype("int32")
                cv2.line(floor_img, pt1, pt2, (45), 2)

            fig.add_subplot(num_floors, 1, n+1)  # type: ignore
            plt.imshow(floor_img)

        if save:
            # cv2.imwrite("data/tmp/" + name + '.png', img)
            plt.savefig(self.graph_path + "/" + name)
        else:
            plt.show()


if __name__ == '__main__':
    # shg_planner = SHGPlanner("data/graphs/benchmarks", "graph.pickle", False)
    shg_planner = SHGPlanner("data/graphs/simulation", "graph.pickle", False)
    # shg_planner = SHGPlanner("data/graphs/iras", "graph.pickle", False)

    # path_dict, distance = shg_planner._plan(["ryu", "room_8", "(1418, 90)"], ["hou2", "room_17", "(186, 505)"])
    # path_dict, distance = shg_planner._plan(["aws1", "room_7", (136, 194)], ["aws1", 'room_7', (156, 144)])
    # path_dict, distance = shg_planner._plan(["aws1", "room_7", (143, 196)], ["aws1", 'room_20', (180, 240)])
    path_dict, distance = shg_planner._plan(['aws1', 'room_7', (163, 246)], ['aws2', 'room_20', (142, 190)])
    ryu_path = shg_planner.get_path_on_floor(["aws1"], key="position", interpolation_resolution=None)
    # hou2_path = shg_planner.get_path_on_floor(["hou2"], key="position", interpolation_resolution=10)
    print("Final path length:", distance, "n:", len(ryu_path))
    # print(len(hou2_path))

    shg_planner.draw_path(save=False, name="path.png")

    G = shg_planner.graph
    floor_ryu = G._get_child("aws1")
    # floor_hou2 = G._get_child("hou2")
    ryu_room_8 = floor_ryu._get_child("room_20")
    # ryu_room_17 = floor_ryu._get_child("room_17")
    # hou2_room_17 = floor_hou2._get_child("room_17")
    # hou2_room_13 = floor_hou2._get_child("room_13")

    # vis.draw_child_graph(G, path_dict, is_leaf=False)
    # vis.draw_child_graph(floor_ryu, path_dict)
    # vis.draw_child_graph(floor_hou2, path_dict)
    vis.draw_child_graph(ryu_room_8, path_dict)
    # vis.draw_child_graph(hou2_room_17, path_dict)
    # vis.draw_child_graph_3d(G, path_dict, is_leaf=False)
    # vis.draw_child_graph_3d(floor_ryu, path_dict)
    # vis.draw_child_graph_3d(floor_hou2, path_dict)
    # vis.draw_child_graph_3d(ryu_room_17, path_dict)
    # vis.draw_child_graph_3d(hou2_room_13, path_dict)
