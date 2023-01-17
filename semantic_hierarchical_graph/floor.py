from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from semantic_hierarchical_graph.environment import Environment
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.node import SHNode
from semantic_hierarchical_graph.parameters import Parameter
import semantic_hierarchical_graph.visualization as vis
import semantic_hierarchical_graph.segmentation as segmentation
import semantic_hierarchical_graph.path_planning as path_planning


class Floor(SHNode):
    def __init__(self, unique_name: str, parent_node, pos: Tuple[float, float, float], map_path: str, params_path: str):
        super().__init__(unique_name, parent_node, pos, False, False)
        self.map = cv2.imread(map_path)
        self.watershed: np.ndarray = np.array([])
        self.params: Dict[str, Any] = Parameter(params_path).params
        self.rooms: Dict[int, Room] = self.create_rooms()

    def segment_map(self):
        self.watershed, ws_erosion, dist_transform = segmentation.marker_controlled_watershed(self.map, self.params)
        bridge_nodes, bridge_edges = segmentation.find_bridge_nodes(self.watershed, dist_transform)
        return ws_erosion, dist_transform, bridge_nodes, bridge_edges

    def create_rooms(self):
        ws_erosion, dist_transform, bridge_nodes, bridge_edges = self.segment_map()
        ws_tmp = ws_erosion.copy()
        rooms: Dict[int, Room] = {}
        # TODO: add rooms to graph instead of dict
        for i in range(2, ws_tmp.max() + 1):
            room_bridge_nodes = {adj_rooms: points for adj_rooms, points in bridge_nodes.items() if i in adj_rooms}
            room_bridge_edges = {adj_rooms: points for adj_rooms, points in bridge_edges.items() if i in adj_rooms}
            rooms[i] = Room(i, self, ws_tmp, self.params, room_bridge_nodes, room_bridge_edges)
        return rooms

    def plot_all_envs(self):
        all_envs = Environment(-1)
        for room in self.rooms.values():
            [all_envs.add_obstacle(obstacle) for obstacle in room.env.scene]
            [all_envs.add_path(path) for path in room.env.path]
        all_envs.plot()

    def draw_all_paths(self, img: np.ndarray,  color) -> np.ndarray:
        img_new = img.copy()
        all_envs = Environment(-1)
        for room in self.rooms.values():
            [cv2.polylines(img_new, [line.coords._coords.astype("int32")], False,  color, 2) for line in room.env.path]

        return img_new

    def get_bridge_nodes(self):
        all_bridge_nodes: Dict[Tuple, List] = {}
        [all_bridge_nodes.update(room.bridge_nodes) for room in self.rooms.values()]
        return all_bridge_nodes

    def get_largest_rectangles(self):
        all_largest_rectangles: Dict[int, List] = {room.id: room.largest_rectangles for room in self.rooms.values()}
        return all_largest_rectangles


class Room(SHNode):
    def __init__(self, id: int, parent_node, ws_erosion: np.ndarray, params: Dict[str, Any], bridge_nodes: Dict[Tuple, List], bridge_edges: Dict[Tuple, List]):
        self.id = id
        self.params: Dict[str, Any] = params
        self.env: Environment = Environment(id)
        self.bridge_nodes: Dict[Tuple, List] = bridge_nodes
        self.largest_rectangles, centroid = path_planning.calc_largest_rectangles(
            ws_erosion, self.env, self.params)
        super().__init__(f"room_{id}", parent_node, centroid, False, False)

        path_planning.connect_paths(self.env, bridge_nodes, bridge_edges)


if __name__ == "__main__":
    G = SHGraph(root_name="LTC Campus", root_pos=(0, 0, 0))

    floor = Floor("ryu", G, (0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")

    G.add_child_node(floor)
    print(G.get_childs("name"))
    vis.draw_child_graph(G, [])
    # vis.draw_graph_3d(G.leaf_graph)

    floor.plot_all_envs()
    ws2 = segmentation.draw(floor.watershed, floor.get_bridge_nodes(), (22))
    # ws3 = segmentation.draw(ws2, floor.get_largest_rectangles(), (21))
    ws4 = floor.draw_all_paths(ws2, (25))
    segmentation.show_imgs(ws4, name="map_benchmark_ryu_result", save=False)
