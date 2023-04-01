import os
from typing import Dict, List, Tuple

from semantic_hierarchical_graph import segmentation, visualization as vis
from semantic_hierarchical_graph.floor import Floor
from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.path import SHPath
from semantic_hierarchical_graph.types.parameter import Parameter
from semantic_hierarchical_graph.types.position import Position


class SHGPlanner():
    def __init__(self, graph_path: str, graph_name: str = "graph.pickle", force_create: bool = False):
        self.graph_path = graph_path
        self.graph: SHGraph = self._init_graph(graph_name, force_create)
        self.path: Dict = {}
        self.distance: float = 0.0

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
            floor = Floor(floor_name, G, Position(0, 0, floor_nr), floor_path + ".png", floor_path + ".yaml")
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
        # TODO: plan from arbitary point and add point to graph
        self.path, self.distance = self.graph.plan_recursive(start, goal)
        SHPath.save_path(self.path, self.graph_path + "/path.json")

        return self.path, self.distance

    def _get_path_on_floor(self, hierarchy_to_floor, only_names, path) -> List:
        leaf_path = []
        for node, dict in path.items():
            if len(hierarchy_to_floor) > 0 and node.unique_name != hierarchy_to_floor[0]:
                continue
            if node.is_leaf and not "bridge" in node.unique_name:
                leaf_path.append(node.unique_name if only_names else node)
            else:
                leaf_path.extend(self._get_path_on_floor(hierarchy_to_floor[1:], only_names, dict))
        return leaf_path

    def get_path_on_floor(self, hierarchy_to_floor, only_names) -> List:
        if self.path is None:
            print("No path found yet. Call plan() first.")
            return []

        path_list = self._get_path_on_floor(hierarchy_to_floor, only_names, self.path)

        # TODO: remove duplicates in node list
        # result = []
        # [result.append(node) for node in path_list if node not in result]

        return list(dict.fromkeys(path_list))

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

            path_list = self._get_path_on_floor(floor.hierarchy, False, self.path)
            for i in range(len(path_list) - 1):
                pt1 = np.round(path_list[i].pos.xy).astype("int32")
                pt2 = np.round(path_list[i + 1].pos.xy).astype("int32")
                cv2.line(floor_img, pt1, pt2, (25), 2)

            fig.add_subplot(num_floors, 1, n+1)  # type: ignore
            plt.imshow(floor_img)

        if save:
            # cv2.imwrite("data/tmp/" + name + '.png', img)
            plt.savefig(self.graph_path + "/" + name)
        else:
            plt.show()


if __name__ == '__main__':
    shg_planner = SHGPlanner("data/graphs/benchmarks", "graph.pickle", False)

    path_dict, distance = shg_planner.plan(["ryu", "room_8", "(1418, 90)"], ["hou2", "room_17", "(186, 505)"])
    ryu_path = shg_planner.get_path_on_floor(["ryu"], only_names=True)
    hou2_path = shg_planner.get_path_on_floor(["hou2"], only_names=True)
    print(len(ryu_path))
    print(len(hou2_path))

    shg_planner.draw_path(save=False, name="path.png")

    G = shg_planner.graph
    floor_ryu = G._get_child("ryu")
    floor_hou2 = G._get_child("hou2")
    ryu_room_8 = floor_ryu._get_child("room_8")
    ryu_room_17 = floor_ryu._get_child("room_17")
    hou2_room_17 = floor_hou2._get_child("room_17")
    hou2_room_13 = floor_hou2._get_child("room_13")

    vis.draw_child_graph(G, path_dict, is_leaf=False)
    vis.draw_child_graph(floor_ryu, path_dict)
    vis.draw_child_graph(floor_hou2, path_dict)
    vis.draw_child_graph(ryu_room_8, path_dict)
    vis.draw_child_graph(hou2_room_17, path_dict)
    vis.draw_child_graph_3d(G, path_dict, is_leaf=False)
    vis.draw_child_graph_3d(floor_ryu, path_dict)
    vis.draw_child_graph_3d(floor_hou2, path_dict)
    vis.draw_child_graph_3d(ryu_room_17, path_dict)
    vis.draw_child_graph_3d(hou2_room_13, path_dict)
