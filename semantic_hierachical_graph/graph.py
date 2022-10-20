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
    'floor': 2,
    'room': 3,
    'location': 4,
    'gridmap': 5,
}

# 3 Alternatives:
# 1. One single graph with all levels
# 2. Hierachical graphs
# 3. Dictionary like structure G["building F"]["floor 1"]["room 1"]

T = TypeVar('T')


class SHNode(Generic[T]):
    def __init__(self, unique_name: str, is_root: bool = False, is_leaf: bool = False):
        self.unique_name = unique_name
        self.is_root = is_root
        self.is_leaf = is_leaf
        self.parent_node: SHNode
        self.child_graph: nx.Graph = nx.Graph()

    def add_child(self, name: str, pos: Tuple[float, float, float], is_leaf: bool = False, **data):
        # Child of type SHNode with unique name on that level

        self.child_graph.add_node(SHNode(unique_name=name, is_leaf=is_leaf), name=name, pos=pos, **data)

    def add_connection(self, child_name_1: str, child_name_2: str, **data):
        self.child_graph.add_edge(self.get_child(child_name_1), self.get_child(child_name_2), **data)

    def get_child(self, name: str):
        for node in self.child_graph.nodes:  # type: ignore
            if node.unique_name == name:
                return node
        return None

    def get_childs(self, key=None):
        if key is None:
            return self.child_graph.nodes()
        else:
            return [value for n, value in self.child_graph.nodes(data=key)]

    def get_dict(self):
        s = {}
        for n, data in self.child_graph.nodes(data=True):
            s[data["name"]] = n.get_dict()  # type: ignore
        return s

    def draw_child_graph(self):
        pos_dict = nx.get_node_attributes(self.child_graph, 'pos')
        for node, pos in pos_dict.items():
            pos_dict[node] = pos[0:2]

        nx.draw(self.child_graph,
                pos=pos_dict,
                labels=nx.get_node_attributes(self.child_graph, 'name'),
                with_labels=True)

        plt.show()


class SHGraph():
    def __init__(self, levels: int, root_name: str, root_pos: Tuple[float, float, float]):
        self.levels = levels
        self.root_node: SHNode = SHNode(unique_name=root_name, is_root=True)

    def add_node(self, level: int, name: str, parent: str, pos: Tuple[float, float, float], **data):
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
    # G.add_node(nx.Graph(), name="Building F", level=1, parent=G, pos=(1, 1, 0))
    # G.add_node(nx.Graph(), name="Building A", level=1, parent=G, pos=(1, 0, 0))
    # print([data["name"] for n, data in G.nodes(data=True)])

    SHG = SHGraph(levels=5, root_name="LTC Campus", root_pos=(0, 0, 0))
    SHG.add_node(level=0, name="Building F", parent="LTC Campus", pos=(1, 1, 0))

    G = SHNode(unique_name="LTC Campus", is_root=True)

    G.add_child(name="Building F", pos=(1, 1, 0))
    G.add_child(name="Building A", pos=(1, 0, 0))
    G.add_connection("Building F", "Building A", name="path_1", type="path")
    print(G.get_dict())
    print(G.get_childs("name"))
    G.draw_child_graph()


if __name__ == "__main__":
    main()
