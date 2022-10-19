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


class SHGraph():
    g: List[nx.Graph] = []

    def __init__(self, hierachy_level: int) -> None:
        self.g = [nx.Graph() for i in range(hierachy_level)]

    def add_connection(self, level: int, node_1: str, node_2: str):
        self.g[0].add_edge(self.get_unique_node_name(level, node_1),
                           self.get_unique_node_name(level, node_2),
                           euclid_distance=self.get_euclidean_distance(
            self.g[0].nodes[self.get_unique_node_name(level, node_1)]['pos'],
            self.g[0].nodes[self.get_unique_node_name(level, node_2)]['pos']))

    def get_euclidean_distance(self, pos_1, pos_2):
        return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2) ** 0.5

    def add_node(self, level: int, pos: Tuple, label: str):
        self.g[level].add_node(self.get_unique_node_name(level, label), pos=pos, label=label)

    def get_unique_node_name(self, level: int, label: str):
        return str(level) + '_' + label

    def draw_single_level(self, level: int):
        nx.draw(self.g[level],
                pos=nx.get_node_attributes(self.g[0], 'pos'),
                labels=nx.get_node_attributes(self.g[0], 'label'),
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
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        fig.tight_layout()
        plt.show()


def main():

    G = SHGraph(3)
    # G.g.add_node('floor_1', _child_graph=floor_1)
    # G.g.add_node('floor_2', child_graph=floor_2)
    G.add_node(0, pos=(1, 1), label='corridor')
    G.add_node(0, pos=(2, 1), label="room_1")
    G.add_node(0, pos=(0, 1), label="room_2")
    # floor_1.add_node('elevator')
    G.add_connection(0, 'corridor', 'room_1')
    G.add_connection(0, 'corridor', 'room_2')
    # G.g.add_edge('floor_1', 'floor_2', type="elevator")

    print(G.g[0].number_of_nodes())
    print(G.g[0].nodes)
    print(G.g[0].edges)

    # G.g.add_nodes_from(floor_1.g)

    G.draw_single_level(level=0)
    G.draw_multiple_level()


if __name__ == "__main__":
    main()
