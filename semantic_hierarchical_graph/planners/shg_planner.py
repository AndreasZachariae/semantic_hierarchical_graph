from collections import deque
import os
from typing import Dict, List, Tuple
import numpy as np
from shapely import LineString, Point
import math

from semantic_hierarchical_graph import roadmap_creation, segmentation, visualization as vis
from semantic_hierarchical_graph.floor import Floor, Room
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.path import SHPath
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError, SHGPlannerError
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
        self.tmp_path_added = []

    def _init_graph(self, graph_name: str, force_create: bool) -> SHGraph:
        if not force_create and os.path.isfile(self.graph_path + "/" + graph_name):
            return self._load_graph(graph_name)

        return self._create_graph()

    def _load_graph(self, graph_name: str) -> SHGraph:
        return SHGraph.load_graph(self.graph_path + "/" + graph_name)

    def _create_graph(self) -> SHGraph:
        params = Parameter(self.graph_path + "/graph.yaml", is_floor=False).params
        G = SHGraph(root_name=self.graph_path.split("/")[-1], root_pos=Position(0, 0, 0), params=params)

        floor_names = {floor_file.split(".")[0] for floor_file in os.listdir(
            self.graph_path + "/" + params["hierarchy_level"][-3])}
        for floor_nr, floor_name in enumerate(floor_names):
            floor_path = os.path.join(self.graph_path, params["hierarchy_level"][-3], floor_name)
            print("Creating floor: " + floor_name)
            floor = Floor(floor_name, G, Position(0, 0, floor_nr), floor_path + ".pgm", floor_path + ".yaml")
            G.add_child_by_node(floor)

            floor.create_rooms()
            floor.create_bridges()

        self._load_connections(G)

        G.save_graph(self.graph_path + "/graph.pickle")

        return G

    def _load_connections(self, graph: SHGraph):
        for connection in graph.params["connections"]:
            graph.add_connection_recursive(connection[0], connection[1], distance=10.0, name="elevator")
            print(connection)

    def add_location(self, hierarchy, location):
        # TODO: add location to graph and save graph
        pass

    def plan(self, start, goal) -> Tuple[Dict, float]:
        print("Plan from " + str(start) + " to " + str(goal))
        start_pos = Position.from_iter(start[-1])
        goal_pos = Position.from_iter(goal[-1])
        start[-1] = start_pos.to_name()
        goal[-1] = goal_pos.to_name()
        start_room: Room = self.graph.get_child_by_hierarchy(start[:-1])  # type: ignore
        goal_room: Room = self.graph.get_child_by_hierarchy(goal[:-1])  # type: ignore
        try:
            distance_to_roadmap = 0.0
            if start_pos.to_name() not in start_room.get_childs("name"):
                distance_to_roadmap += self._add_path_to_roadmap(start_room,
                                                                 start_pos.to_name(), start_pos, type="start")
            if goal_pos.to_name() not in goal_room.get_childs("name"):
                distance_to_roadmap += self._add_path_to_roadmap(goal_room, goal_pos.to_name(), goal_pos, type="goal")

            if start_room == goal_room:
                self.path, distance = self._check_for_direct_connection(
                    distance_to_roadmap, start_room, start_pos, goal_pos)
                if self.path is not None:
                    return self.path, distance

            self.path, self.distance = self.graph.plan_recursive(start, goal)
            # vis.draw_child_graph(start_room, self.path)
        except SHGPlannerError as e:
            print("Error while planning with SHGPlanner: ")
            print(e)
        finally:
            self._revert_tmp_graph(start_room)
            self._revert_tmp_graph(goal_room)
            self.tmp_edge_removed = []
            self.tmp_path_added = []

        SHPath.save_path(self.path, self.graph_path + "/path.json")

        return self.path, self.distance

    def _check_for_direct_connection(self, distance_to_roadmap: float, start_room: Room, start_pos: Position, goal_pos: Position) -> Tuple:
        if distance_to_roadmap > start_pos.distance(goal_pos):
            connection = start_room.env.get_valid_connection(Point(start_pos.xy), Point(goal_pos.xy))
            if connection is not None:
                print("Start point is closer to goal than to roadmap")
                path = {start_room.parent_node: {start_room:
                                                 {start_room._get_child(start_pos.to_name()): {},
                                                  start_room._get_child(goal_pos.to_name()): {}}}}
                return path, start_pos.distance(goal_pos)

        return None, 0

    def _add_path_to_roadmap(self, room_node: Room, node_name, node_pos, type) -> float:
        if room_node.env._in_collision(Point(node_pos.xy)):
            raise SHGPlannerError(
                f"Point {node_pos.xy} is not in the drivable area (boundary + safety margin) of the room")

        connections, closest_path = roadmap_creation.connect_point_to_path(node_pos.xy, room_node.env, room_node.params)

        if len(connections) == 0 or closest_path is None:
            raise SHGPlannerError(f"No connection from point {node_pos.xy} to roadmap found")

        nodes = deque(maxlen=2)
        nodes.append(room_node.add_child_by_name(node_name,
                                                 Position.from_iter(connections[0].coords[0]),
                                                 True, type="aux_node"))
        distance = 0.0
        for connection in connections:
            pos = Position.from_iter(connection.coords[1])
            nodes.append(room_node.add_child_by_name(pos.to_name(), pos, True, type="aux_node"))
            d = nodes[0].pos.distance(nodes[1].pos)
            distance += d
            room_node.add_connection_by_nodes(nodes[0], nodes[1], d)

        if len(closest_path.coords) != 2:
            raise SHGGeometryError("Path has not 2 points as expected")

        path_1_pos = Position.from_iter(closest_path.coords[0])
        path_2_pos = Position.from_iter(closest_path.coords[1])
        path_1_node = room_node._get_child(path_1_pos.to_name())
        path_2_node = room_node._get_child(path_2_pos.to_name())
        if room_node.child_graph.has_edge(path_1_node, path_2_node):
            edge_data = room_node.child_graph.get_edge_data(path_1_node, path_2_node)
            # print(f"Removing edge {edge_data} from roadmap")
            room_node.child_graph.remove_edge(path_1_node, path_2_node)
            room_node.env.path.remove(closest_path)
            self.tmp_edge_removed.append((path_1_node, path_2_node, edge_data, closest_path))
        room_node.add_connection_by_nodes(path_1_node, nodes[1], path_1_pos.distance(nodes[1].pos))
        # print(f"Added edge {path_1_node.unique_name, nodes[1].unique_name} to roadmap")
        room_node.add_connection_by_nodes(nodes[1], path_2_node, path_2_pos.distance(nodes[1].pos))
        # print(f"Added edge {nodes[1].unique_name, path_2_node.unique_name} to roadmap")
        path1 = LineString([path_1_pos.xy, nodes[1].pos.xy])
        path2 = LineString([nodes[1].pos.xy, path_2_pos.xy])
        self.tmp_path_added.append(path1)
        self.tmp_path_added.append(path2)
        room_node.env.add_path(path1)
        room_node.env.add_path(path2)

        return distance

    def _revert_tmp_graph(self, room_node):
        to_remove = []
        for node in room_node.get_childs():
            if node.type == "aux_node":
                to_remove.append(node)
        for node in to_remove:
            room_node.child_graph.remove_node(node)

        for edge in list(self.tmp_edge_removed):
            if edge[0] in room_node.child_graph.nodes and edge[1] in room_node.child_graph.nodes:
                room_node.child_graph.add_edge(edge[0], edge[1], **edge[2])
                room_node.env.add_path(edge[3])
                # print(f"Added edge {edge[2]} to roadmap")
                self.tmp_edge_removed.remove(edge)

        for path in list(self.tmp_path_added):
            if path in room_node.env.path:
                room_node.env.path.remove(path)
                self.tmp_path_added.remove(path)

        # print(f"Removed {len(to_remove)} nodes for cleanup")

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
        if self.path is None:
            print("No path found yet. Call plan() first.")
            return []

        path_list: List[Position] = self._get_path_on_floor(hierarchy_to_floor, key, self.path)
        path_list = list(dict.fromkeys(path_list))  # remove duplicates

        for i in range(len(path_list)-1):
            if path_list[i].rz is None:
                angle = Vector.from_two_points(path_list[i].xy, path_list[i+1].xy).angle_to_grid()
                path_list[i].rz = angle
                # print(path_list[i].xy, path_list[i+1].xy, angle)

        if interpolation_resolution is not None:
            print("Interpolating path with resolution", interpolation_resolution, "px")
            path_list = self._interpolate_path(path_list, interpolation_resolution)

        # [print(node.xy, node.rz) for node in path_list]

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
                pt1 = np.round(path_list[i].pos.xy).astype("int32")
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

    # path_dict, distance = shg_planner.plan(["ryu", "room_8", "(1418, 90)"], ["hou2", "room_17", "(186, 505)"])
    # path_dict, distance = shg_planner.plan(["aws1", "room_7", (136, 194)], ["aws1", 'room_7', (156, 144)])
    path_dict, distance = shg_planner.plan(["aws1", "room_7", (143, 196)], ["aws1", 'room_20', (180, 240)])
    ryu_path = shg_planner.get_path_on_floor(["aws1"], key="position", interpolation_resolution=10)
    # hou2_path = shg_planner.get_path_on_floor(["hou2"], key="position", interpolation_resolution=10)
    print("Final path length:", distance, "n:", len(ryu_path))
    # print(len(hou2_path))

    shg_planner.draw_path(save=False, name="path.png")

    G = shg_planner.graph
    floor_ryu = G._get_child("aws1")
    # floor_hou2 = G._get_child("hou2")
    ryu_room_8 = floor_ryu._get_child("room_7")
    # ryu_room_17 = floor_ryu._get_child("room_17")
    # hou2_room_17 = floor_hou2._get_child("room_17")
    # hou2_room_13 = floor_hou2._get_child("room_13")

    vis.draw_child_graph(G, path_dict, is_leaf=False)
    vis.draw_child_graph(floor_ryu, path_dict)
    # vis.draw_child_graph(floor_hou2, path_dict)
    vis.draw_child_graph(ryu_room_8, path_dict)
    # vis.draw_child_graph(hou2_room_17, path_dict)
    # vis.draw_child_graph_3d(G, path_dict, is_leaf=False)
    # vis.draw_child_graph_3d(floor_ryu, path_dict)
    # vis.draw_child_graph_3d(floor_hou2, path_dict)
    # vis.draw_child_graph_3d(ryu_room_17, path_dict)
    # vis.draw_child_graph_3d(hou2_room_13, path_dict)
