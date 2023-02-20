from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import semantic_hierarchical_graph.utils as util
from semantic_hierarchical_graph.metrics import Metrics
from semantic_hierarchical_graph.environment import Environment
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.node import SHNode
from semantic_hierarchical_graph.parameters import Parameter
import semantic_hierarchical_graph.visualization as vis
import semantic_hierarchical_graph.segmentation as segmentation
import semantic_hierarchical_graph.path_planning as path_planning
from semantic_hierarchical_graph.types import Position


class Floor(SHNode):
    def __init__(self, unique_name: str, parent_node, pos: Position, map_path: str, params_path: str):
        super().__init__(unique_name, parent_node, pos, False, False)
        self.map = cv2.imread(map_path)
        self.watershed: np.ndarray = np.array([])
        self.dist_transform: np.ndarray = np.array([])
        self.params: Dict[str, Any] = Parameter(params_path).params
        self.all_bridge_nodes: Dict[Tuple, List] = {}
        self.bridge_points_not_connected: Set = set()

    def create_rooms(self):
        self.watershed, ws_erosion, self.dist_transform = segmentation.marker_controlled_watershed(
            self.map, self.params)
        self.all_bridge_nodes, bridge_edges = segmentation.find_bridge_nodes(ws_erosion, self.dist_transform)
        ws_tmp = ws_erosion.copy()
        for i in range(2, ws_tmp.max() + 1):
            room_bridge_nodes = {adj_rooms: points for adj_rooms, points in self.all_bridge_nodes.items()
                                 if i in adj_rooms}
            room_bridge_edges = {adj_rooms: edges for adj_rooms, edges in bridge_edges.items()
                                 if i in adj_rooms}
            room_mask = np.where(self.watershed == i, 255, 0).astype("uint8")
            room = Room(i, self, ws_tmp, room_mask, self.params, room_bridge_nodes, room_bridge_edges)
            self.bridge_points_not_connected.update(room.bridge_points_not_connected)
            self.add_child_by_node(room)
            room.create_roadmap()

    def create_bridges(self):
        for adj_rooms, bridge_points in self.all_bridge_nodes.items():
            for point in bridge_points:
                if point in self.bridge_points_not_connected:
                    continue
                h1 = self.hierarchy + [f"room_{adj_rooms[0]}", str(point)]
                h2 = self.hierarchy + [f"room_{adj_rooms[1]}", str(point)]
                self._get_child(f"room_{adj_rooms[0]}")._get_child(str(point))
                self._get_root_node().add_connection_recursive(h1, h2, distance=0.0)

    def plot_all_envs(self):
        all_envs = Environment(-1)
        for room in self.get_childs():
            [all_envs.add_obstacle(obstacle) for obstacle in room.env.scene]
            [all_envs.add_path(path) for path in room.env.path]
        print("All paths:", len(all_envs.path))
        all_envs.plot()

    def draw_all_paths(self, img: np.ndarray,  color, path: Optional[Dict] = None, path_color=None) -> np.ndarray:
        img_new = img.copy()
        for room in self.get_childs():
            [cv2.polylines(img_new, [line.coords._coords.astype("int32")], False,  color, 2) for line in room.env.path]
        if path is not None:
            path_list = util._path_dict_to_leaf_path_list(path)
            for i in range(len(path_list) - 1):
                pt1 = np.round(path_list[i].pos.xy).astype("int32")
                pt2 = np.round(path_list[i + 1].pos.xy).astype("int32")
                cv2.line(img_new, pt1, pt2, path_color, 2)

        return img_new

    def get_largest_rectangles(self):
        all_largest_rectangles: Dict[int, List] = {room.id: room.largest_rectangles for room in self.get_childs()}
        return all_largest_rectangles


class Room(SHNode):
    def __init__(self, id: int, parent_node, ws_erosion: np.ndarray, mask: np.ndarray, params: Dict[str, Any], bridge_nodes: Dict[Tuple, List], bridge_edges: Dict[Tuple, List]):
        self.id = id
        self.mask = mask
        self.params: Dict[str, Any] = params
        self.env: Environment = Environment(id)
        self.bridge_nodes: Dict[Tuple, List] = bridge_nodes
        self.largest_rectangles, centroid = path_planning.calc_largest_rectangles(
            ws_erosion, self.env, self.params)
        super().__init__(f"room_{id}", parent_node, centroid, False, False)

        self.bridge_points_not_connected: Set = path_planning.connect_paths(
            self.env, bridge_nodes, bridge_edges, self.params)

    def create_roadmap(self):
        self.env.remove_duplicate_paths()
        self.env.split_multipoint_lines()
        self.env.split_path_at_intersections()
        # print(len(self.env.path))

        for path in self.env.path:
            if len(path.coords) != 2:
                raise ValueError("Path has not 2 points as expected")
            connection = []
            for point in path.coords:
                pos = Position.from_iter(point)
                name = pos.to_name()
                if name in self.get_childs("name"):
                    loc = self._get_child(name)
                else:
                    loc = Location(name, self, pos)
                    self.add_child_by_node(loc)
                connection.append(loc)
            self.add_connection_by_nodes(connection[0], connection[1])
        # self.env.plot()


class Location(SHNode):
    def __init__(self, unique_name: str, parent_node, pos: Position):
        super().__init__(unique_name, parent_node, pos, False, True)


if __name__ == "__main__":
    G = SHGraph(root_name="Benchmark", root_pos=Position(0, 0, 0))
    floor = Floor("ryu", G, Position(0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")
    G.add_child_by_node(floor)
    print(G.get_childs("name"))

    floor.create_rooms()
    floor.create_bridges()

    # print(floor.get_childs("name"))
    room_2 = floor._get_child("room_2")
    room_11 = floor._get_child("room_11")
    # print(room_2.get_childs("name"))

    path_dict = G.plan_recursive(["ryu", "room_2", "(387, 60)"], ["ryu", "room_12", "(1526, 480)"])
    # util.save_dict_to_json(path_dict, "data/ryu_path.json")

    # vis.draw_child_graph(floor, path_dict)
    # vis.draw_child_graph(room_2, path_dict)
    vis.draw_child_graph(G, path_dict, is_leaf=True)
    vis.draw_child_graph_3d(floor, path_dict)
    vis.draw_child_graph_3d(room_11, path_dict)
    # vis.draw_child_graph_3d(G, path_dict, is_leaf=True)

    floor.plot_all_envs()
    ws2 = segmentation.draw(floor.watershed, floor.all_bridge_nodes, (22))
    # ws3 = segmentation.draw(ws2, floor.get_largest_rectangles(), (21))
    ws4 = floor.draw_all_paths(ws2, (0), path_dict, (25))
    segmentation.show_imgs(ws4, name="map_benchmark_ryu_result", save=False)

    metrics = Metrics(room_11)
    # metrics.print_metrics()
    metrics.save_metrics("data/ryu_metrics.json")
