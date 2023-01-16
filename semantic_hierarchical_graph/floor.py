from typing import Tuple

import cv2
import numpy as np
from semantic_hierarchical_graph.environment import Environment
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.node import SHNode
from semantic_hierarchical_graph.parameters import Parameter
import semantic_hierarchical_graph.visualization as vis
import semantic_hierarchical_graph.segmentation as segmentation


class Floor(SHNode):
    def __init__(self, unique_name: str, parent_node, pos: Tuple[float, float, float], map_path: str, params_path: str):
        super().__init__(unique_name, parent_node, pos, False, False)
        self.map = cv2.imread(map_path)
        self.params = Parameter(params_path).params
        self.rooms: dict[int, Room] = {}

    def segment_map(self):
        ws, ws_erosion, dist_transform = segmentation.marker_controlled_watershed(self.map, self.params)
        bridge_nodes, bridge_edges = segmentation.find_bridge_nodes(ws, dist_transform)
        return ws, ws_erosion, dist_transform, bridge_nodes, bridge_edges

    def create_rooms(self):
        ws, ws_erosion, dist_transform, bridge_nodes, bridge_edges = self.segment_map()
        for i in range(2, ws.max() + 1):
            segment_bool = np.where(ws == i, True, False)
            M = cv2.moments(segment_bool)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            self.rooms[i] = Room(f"room_{i}", self, (cX, cY, 0))

    def create_connections(self):
        pass


class Room(SHNode):
    def __init__(self, unique_name: str, parent_node, pos: Tuple[float, float, float]):
        super().__init__(unique_name, parent_node, pos, False, False)
        self.env = Environment()
        self.largest_rectangles: list = []


if __name__ == "__main__":
    G = SHGraph(root_name="LTC Campus", root_pos=(0, 0, 0))

    ryu_floor = Floor("ryu", G, (0, 0, 1), 'data/ryu.png', "config/ryu_params.yaml")

    G.add_child_node(ryu_floor)
    print(G.get_childs("name"))
    vis.draw_child_graph(G, [])
    vis.draw_graph_3d(G.leaf_graph)
