import networkx as nx
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
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


class TreeNode:
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.children = []

    def addChild(self, child):
        self.children.append(child)

    def serialize(self):
        s = {}
        for child in self.children:
            s[child.name] = child.serialize()
        return s


class Level():
    def __init__(self, parent: nx.Graph):
        self.graph = nx.Graph()
        self.parent: nx.Graph = parent


class SHGraph():
    g: List[nx.Graph] = []

    def __init__(self, hierachy_level: int) -> None:
        self.g = [nx.Graph() for i in range(hierachy_level)]

    def add_connection(self, level: int, label_1: str, label_2: str):
        self.g[level].add_edge(self.get_unique_name(level, label_1),
                               self.get_unique_name(level, label_2),
                               euclid_distance=self.get_euclidean_distance(
            self.g[level].nodes[self.get_unique_name(level, label_1)]['pos'],
            self.g[level].nodes[self.get_unique_name(level, label_2)]['pos']))

    def get_euclidean_distance(self, pos_1: Tuple, pos_2: Tuple) -> float:
        return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2) ** 0.5

    def add_node(self, level: int, parent: str, pos: Tuple, label: str):
        self.g[level].add_node(self.get_unique_name(level, parent, label), pos=pos, label=label)

    def get_unique_name(self, level: int, parent: str, label: str) -> str:
        if parent:
            return f'{self.g[level-1]["label"]}_{label}'
        else:
            return f'{label}'

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


def main():

    G = SHGraph(3)
    G.add_node(level=0, parent=None, pos=(0, 0, 0), label='LTC Campus')
    G.add_node(level=0, parent=pos=(1, 1, 1), label='corridor')
    G.add_node(level=0, pos=(2, 1, 1), label="room_1")
    G.add_node(level=0, pos=(0, 1, 1), label="room_2")
    # floor_1.add_node('elevator')
    G.add_connection(level=0, label_1='corridor', label_2='room_1')
    G.add_connection(level=0, label_1='corridor', label_2='room_2')
    # G.g.add_edge('floor_1', 'floor_2', type="elevator")

    print(G.g[0].number_of_nodes())
    print(G.g[0].nodes)
    print(G.g[0].edges)

    # G.g.add_nodes_from(floor_1.g)

    G.draw_single_level(level=0)
    G.draw_multiple_level()


if __name__ == "__main__":
    main()
