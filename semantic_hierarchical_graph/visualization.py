from typing import Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from semantic_hierarchical_graph.graph import SHGraph
import semantic_hierarchical_graph.utils as util


# TODO: Update function to draw every graph not just from root node
def draw_child_graph(root_node: SHGraph, child_hierarchy: List[str], path_dict: Optional[Dict] = None, view_axis: int = 2):
    child = root_node.get_child_by_hierarchy(child_hierarchy)

    pos_index = [0, 1, 2]
    pos_index.remove(view_axis)
    pos_dict = {node: (node.pos_abs[pos_index[0]], node.pos_abs[pos_index[1]]) for node in child.get_childs()}

    if path_dict is not None:
        path_list = util.path_dict_to_child_path_list(path_dict, child_hierarchy)
        # print(util.map_names_to_nodes(path_list))
        edge_colors = []
        for u, v in child.child_graph.edges:
            if u in path_list and v in path_list:
                edge_colors.append('red')
            else:
                edge_colors.append(child.child_graph[u][v]["color"])
    else:
        edge_colors = None

    nx.draw(child.child_graph,
            pos=pos_dict,
            edge_color=edge_colors,
            labels=nx.get_node_attributes(child.child_graph, 'name'),
            with_labels=True)

    plt.show()


# TODO: Update function to draw every path directly from dict
def draw_graph_3d(graph: nx.Graph, path=None):
    node_xyz = np.array([node.pos_abs for node in graph.nodes()])  # type: ignore
    edge_xyz = np.array([(u.pos_abs, v.pos_abs) for u, v in graph.edges])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*node_xyz.T, s=100, ec="w")  # type: ignore

    for node in graph.nodes():
        ax.text(node.pos_abs[0], node.pos_abs[1], node.pos_abs[2],  # type: ignore
                node.unique_name, size=9, color='k')  # type: ignore

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


def plot_times():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    type = ['recursive', 'leaf graph']
    times = [85.7, 24.3]
    ax.set_ylabel('Time in μs')
    ax.set_title('Time comparison leaf graph vs. H-Graph')
    ax.bar(type, times)
    fig.tight_layout()
    plt.show()
