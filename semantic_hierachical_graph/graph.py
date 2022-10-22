import networkx as nx
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, TypeVar, Generic
import numpy as np

node_attributes = {
    'label': 'corridor',
}

edge_attributes = {
    'time': 10,
    'euclid_distance': 10,
    'path_distance': 10,
    'weight': 1,
    'label': 'elevator',
    'type': 'elevator'
}

hierarchy_levels = {
    'campus': 0,
    'building': 1,
    'wing': 2,
    'floor': 3,
    'room': 4,
    'location': 5,
    'gridmap': 6
}

# 3 Alternatives:
# 1. One single graph with all levels
# 2. Hierachical graphs
# 3. Dictionary like structure G["building F"]["floor 1"]["room 1"]

T = TypeVar('T', bound='SHNode')


class SHNode(Generic[T]):
    def __init__(self, unique_name: str, parent_node, pos: Tuple[float, float, float], is_root: bool = False, is_leaf: bool = False):
        self.unique_name: str = unique_name
        self.is_root: bool = is_root
        self.is_leaf: bool = is_leaf
        self.pos: Tuple[float, float, float] = pos
        self.pos_abs: Tuple[float, float, float] = tuple(np.add(pos, parent_node.pos_abs)) if not is_root else pos
        self.parent_node: SHNode = parent_node
        self.child_graph: nx.Graph = nx.Graph()

    def _add_child(self, name: str, pos: Tuple[float, float, float], is_leaf: bool = False, **data):
        # Child of type SHNode with unique name on that level

        self.child_graph.add_node(SHNode(unique_name=name, parent_node=self, pos=pos, is_leaf=is_leaf),
                                  name=name, **data)

    def _add_connection(self, child_name_1: str, child_name_2: str, **data):
        self.child_graph.add_edge(self._get_child(child_name_1), self._get_child(child_name_2), **data)

    def _get_child(self, name: str) -> T:
        # Could be way more efficient if using hierarchy_list as hashable node instead of SHNode object
        # Drawback: node.pos etc is not that easy, has to get the data dict all the time graph.nodes(data=True)
        for node in self.child_graph.nodes:
            if node.unique_name == name:
                return node
        raise ValueError("Child with name {} not found".format(name))

    def get_childs(self, key=None) -> List[T]:
        if key is None:
            return self.child_graph.nodes()  # type: ignore
        else:
            return [value for n, value in self.child_graph.nodes(data=key)]  # type: ignore

    def get_dict(self) -> dict:
        s = {}
        for node in self.child_graph.nodes:
            s[node.unique_name] = node.get_dict()
        return s

    def draw_child_graph(self, view_axis: int = 2):
        pos_index = [0, 1, 2]
        pos_index.remove(view_axis)
        pos_dict = {node: (node.pos_abs[pos_index[0]], node.pos_abs[pos_index[1]]) for node in self.get_childs()}

        nx.draw(self.child_graph,
                pos=pos_dict,
                labels=nx.get_node_attributes(self.child_graph, 'name'),
                with_labels=True)

        plt.show()


class SHGraph(SHNode):
    def __init__(self,  root_name: str, root_pos: Tuple[float, float, float]):
        self.root_graph: nx.Graph = nx.Graph()
        self.leaf_graph: nx.Graph = nx.Graph()
        self.root_graph.add_node(self, name=root_name)
        super().__init__(unique_name=root_name,
                         parent_node=self.root_graph[self], pos=root_pos, is_root=True)

    def add_child(self, hierarchy: List[str], name: str, pos: Tuple[float, float, float], is_leaf: bool = False, **data):
        self.get_child(hierarchy)._add_child(name, pos, is_leaf, **data,)
        if is_leaf:
            hierarchy.append(name)
            self.leaf_graph.add_node(self.get_child(hierarchy),
                                  name=name, **data)

    def add_connection(self, hierarchy_1: List[str], hierarchy_2: List[str], **data):
        add_hierarchy_bridge: bool = False
        for index, (name_1, name_2) in enumerate(zip(hierarchy_1, hierarchy_2)):
            if add_hierarchy_bridge:
                print("New hierarchy connection between:")
                print(name_1, hierarchy_2[index-1] + "_h_bridge", "in graph:", hierarchy_1[:index])
                print(name_2, hierarchy_1[index-1] + "_h_bridge", "in graph:", hierarchy_2[:index])
                self.get_child(hierarchy_1[:index])._add_child(hierarchy_2[index-1] + "_h_bridge",pos=(0,0,0), type="hierarchy_bridge")
                self.get_child(hierarchy_1[:index])._add_connection(name_1, hierarchy_2[index-1] + "_h_bridge", **data)
                self.get_child(hierarchy_2[:index])._add_child(hierarchy_1[index-1] + "_h_bridge",pos=(0,0,0), type="hierarchy_bridge")
                self.get_child(hierarchy_2[:index])._add_connection(name_2, hierarchy_1[index-1] + "_h_bridge", **data)
            elif name_1 != name_2:
                print("New connection between:",name_1, name_2, "in graph:", hierarchy_1[:index])
                self.get_child(hierarchy_1[:index])._add_connection(name_1, name_2, **data)
                add_hierarchy_bridge = True
            if name_1 == hierarchy_1[-1]:
                node_1 = self.get_child(hierarchy_1)
                if node_1.is_leaf:
                    self.leaf_graph.add_edge(node_1, self.get_child(hierarchy_2), **data)

    def get_child(self, hierarchy: List[str]) -> SHNode:
        child = self
        for name in hierarchy:
            child = child._get_child(name)
        return child

    def draw_leaf_graph(self):
        node_xyz = np.array([node.pos_abs for node in self.leaf_graph.nodes()])
        edge_xyz = np.array([(u.pos_abs, v.pos_abs) for u, v in self.leaf_graph.edges])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*node_xyz.T, s=100, ec="w")

        for node in self.leaf_graph.nodes():
            ax.text(node.pos_abs[0], node.pos_abs[1], node.pos_abs[2],  node.unique_name, size=7, color='k') 

        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):  # type: ignore
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")  # type: ignore

        fig.tight_layout()
        plt.show()


    def create_graph_from_dict(self):
        pass

    def plan(self, start_hierarchy: List[str], goal_hierarchy: List[str]):
        pass


class TestGraph():
    g: List[nx.Graph] = []

    def __init__(self, hierachy_level: int) -> None:
        self.g = [nx.Graph() for i in range(hierachy_level)]
        self.root_graph = nx.Graph(name="LTC Campus", parent=None, pos=(0, 0, 0))

    def add_connection(self, level: int, label_1: str, label_2: str):
        self.g[level].add_edge(self.get_unique_name(level, label_1),
                               self.get_unique_name(level, label_2),
                               euclid_distance=self.get_euclidean_distance(
            self.g[level].nodes[self.get_unique_name(level, label_1)]['pos'],
            self.g[level].nodes[self.get_unique_name(level, label_2)]['pos']))

    def get_euclidean_distance(self, pos_1: Tuple, pos_2: Tuple) -> float:
        return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2) ** 0.5

    def add_node(self, level: int, pos: Tuple, label: str):
        self.g[level].add_node(self.get_unique_name(level, label), pos=pos, label=label)

    def get_unique_name(self, level: int, label: str) -> str:
        return f'{str(level)}_{label}'

    def draw_single_level(self, level: int):
        pos_dict = nx.get_node_attributes(self.g[level], 'pos')
        for node, pos in pos_dict.items():
            pos_dict[node] = pos[0:2]

        nx.draw(self.g[level],
                pos=pos_dict,
                labels=nx.get_node_attributes(self.g[level], 'label'),
                with_labels=True)

        plt.show()

    def draw_multiple_level(self):
        pos = nx.get_node_attributes(self.g[0], 'pos')
        node_xyz = np.array([pos[v] for v in sorted(self.g[0])])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in self.g[0].edges])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*node_xyz.T, s=100, ec="w")

        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):  # type: ignore
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")  # type: ignore

        fig.tight_layout()
        plt.show()

    def create_example_data(self):

        self.add_node(level=0, pos=(0, 0, 0), label='LTC Campus')
        self.add_node(level=0, pos=(1, 1, 1), label='corridor')
        self.add_node(level=0, pos=(2, 1, 1), label="room_1")
        self.add_node(level=0, pos=(0, 1, 1), label="room_2")
        # floor_1.add_node('elevator')
        self.add_connection(level=0, label_1='corridor', label_2='room_1')
        self.add_connection(level=0, label_1='corridor', label_2='room_2')
        # G.g.add_edge('floor_1', 'floor_2', type="elevator")

        print(self.g[0].number_of_nodes())
        print(self.g[0].nodes)
        print(self.g[0].edges)
        # G.g.add_nodes_from(floor_1.g)


def main():
    # G = TestGraph(3)
    # G.create_example_data()
    # G.draw_single_level(level=0)
    # G.draw_multiple_level()

    # G = nx.Graph(name="LTC Campus", level=0, pos=(0, 0, 0))
    # child = nx.Graph()
    # G.add_node(child, name="Building F", level=1, parent=G, pos=(1, 1, 0))
    # G.add_node(nx.Graph(), name="Building A", level=1, parent=G, pos=(1, 0, 0))
    # print(G.nodes[child])
    # print([data["name"] for n, data in G.nodes(data=True)])

    G = SHGraph(root_name="LTC Campus", root_pos=(0, 0, 0))

    # G._add_child(name="Building F", pos=(-5, 0, 0))
    # G._get_child("Building F")._add_child(name="Floor 0", pos=(0, 0, 0))
    # G._get_child("Building F")._get_child("Floor 0")._add_child(name="Staircase", pos=(0, -1, 0))
    # G.add_connection("Building F", "Building A", name="path_1", type="path")

    G.add_child(hierarchy=[], name="Building A", pos=(0, 0, 0))
    G.add_child(hierarchy=[], name="Building B", pos=(5, 0, 0))
    G.add_child(hierarchy=[], name="Building C", pos=(5, 5, 0))
    G.add_child(hierarchy=[], name="Building D", pos=(0, 5, 0))
    G.add_child(hierarchy=[], name="Building E", pos=(-5, 5, 0))
    G.add_child(hierarchy=[], name="Building F", pos=(-5, 0, 0))
    G.add_child(hierarchy=["Building F"], name="Floor 0", pos=(0, 0, 0))
    G.add_child(hierarchy=["Building F"], name="Floor 1", pos=(0, 0, 1))
    G.add_child(hierarchy=["Building F"], name="Floor 2", pos=(0, 0, 2))
    G.add_child(hierarchy=["Building F"], name="Floor 3", pos=(0, 0, 3))
    G.add_child(hierarchy=["Building A"], name="Floor 0", pos=(0, 0, 0))
    G.add_child(hierarchy=["Building A"], name="Floor 1", pos=(0, 0, 1))
    G.add_child(hierarchy=["Building A"], name="Floor 2", pos=(0, 0, 2))
    G.add_child(hierarchy=["Building A"], name="Floor 3", pos=(0, 0, 3))
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="IRAS", pos=(0, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="xLab", pos=(1, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Kitchen", pos=(2, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Corridor", pos=(1, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Meeting Room", pos=(1, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="Lab", pos=(0, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="Workshop", pos=(1, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="RoboEduLab", pos=(0, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 3"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 3"], name="Office", pos=(1, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building A", "Floor 1"], name="Cantina", pos=(1, 0, 0), is_leaf=True)

    # print(G.get_dict())
    # print(G.get_childs("name"))
    # [print(node.pos, node.pos_abs) for node in G._get_child("Building F").get_childs()]

    G.add_connection(["Building F", "Floor 0", "Staircase"],
                     ["Building F", "Floor 0", "Lab"], name="floor_door", type="door")
    G.add_connection(["Building F", "Floor 0", "Staircase"],
                     ["Building F", "Floor 3", "Staircase"], name="stair_F", type="stair")
    G.add_connection(["Building F", "Floor 1", "Kitchen"],
                     ["Building A", "Floor 1", "Cantina"], name="terrace_door", type="door")

    # G.draw_child_graph()
    # G.get_child(["Building F"]).draw_child_graph(view_axis=0)
    # G.get_child(["Building F", "Floor 0"]).draw_child_graph()
    # G.get_child(["Building F", "Floor 1"]).draw_child_graph()
    # G.get_child(["Building F", "Floor 3"]).draw_child_graph()

    G.draw_leaf_graph()



if __name__ == "__main__":
    main()
