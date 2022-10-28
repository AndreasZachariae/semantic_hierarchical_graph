import networkx as nx
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from semantic_hierarchical_graph.node import SHNode


def draw_child_graph(node: SHNode, view_axis: int = 2):
    pos_index = [0, 1, 2]
    pos_index.remove(view_axis)
    pos_dict = {node: (node.pos_abs[pos_index[0]], node.pos_abs[pos_index[1]]) for node in node.get_childs()}

    nx.draw(node.child_graph,
            pos=pos_dict,
            labels=nx.get_node_attributes(node.child_graph, 'name'),
            with_labels=True)

    plt.show()


def draw_graph_3d(graph: nx.Graph, path=None):
    node_xyz = np.array([node.pos_abs for node in graph.nodes()])  # type: ignore
    edge_xyz = np.array([(u.pos_abs, v.pos_abs) for u, v in graph.edges])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*node_xyz.T, s=100, ec="w")  # type: ignore

    for node in graph.nodes():
        ax.text(node.pos_abs[0], node.pos_abs[1], node.pos_abs[2],  # type: ignore
                node.unique_name, size=7, color='k')  # type: ignore

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    if path is not None:
        path_xyz = np.array([node.pos_abs for node in path])
        ax.plot(*path_xyz.T, color="tab:red")

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