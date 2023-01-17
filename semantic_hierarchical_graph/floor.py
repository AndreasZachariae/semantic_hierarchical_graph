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
            rooms[i] = Room(i, self, ws_tmp, self.params)
        return rooms

    def get_bridge_nodes(self):
        all_bridge_nodes: Dict[Tuple, List] = {}
        [all_bridge_nodes.update(room.bridge_nodes) for room in self.rooms.values()]
        return all_bridge_nodes

    def get_largest_rectangles(self):
        all_largest_rectangles: Dict[int, List] = {room.id: room.largest_rectangles for room in self.rooms.values()}
        return all_largest_rectangles


class Room(SHNode):
    def __init__(self, id: int, parent_node, ws_erosion: np.ndarray, params: Dict[str, Any]):
        self.id = id
        self.params: Dict[str, Any] = params
        self.env: Environment = Environment()
        self.bridge_nodes: Dict[Tuple, List] = {}
        self.largest_rectangles, centroid = path_planning.calc_largest_rectangles(
            ws_erosion, self.id, self.env, self.params)
        super().__init__(f"room_{id}", parent_node, centroid, False, False)


if __name__ == "__main__":
    G = SHGraph(root_name="LTC Campus", root_pos=(0, 0, 0))

    floor = Floor("ryu", G, (0, 0, 1), 'data/benchmark_maps/ryu.png', "config/ryu_params.yaml")

    G.add_child_node(floor)
    print(G.get_childs("name"))
    vis.draw_child_graph(G, [])
    # vis.draw_graph_3d(G.leaf_graph)

    ws2 = segmentation.draw(floor.watershed, floor.get_bridge_nodes(), (22))
    ws3 = segmentation.draw(ws2, floor.get_largest_rectangles(), (21))
    # ws4 = draw_all_paths(ws2, segment_envs, (25))
    segmentation.show_imgs(ws3, name="map_benchmark_ryu_result", save=False)
