# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx


def aStarVisualize(planner, solution, ax=None, nodeSize=300):
    graph = planner.graph
    collChecker = planner._collisionChecker
    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    pos = nx.get_node_attributes(graph, 'pos')
    color = nx.get_node_attributes(graph, 'color')

    # get a list of degrees of all nodes
    #degree = nx.degree_centrality(graph)

    # draw graph (nodes colorized by degree)
    open_nodes = [node for node, attribute in graph.nodes(data=True) if attribute['status'] == "open"]
    draw_nodes = nx.draw_networkx_nodes(graph, pos, node_color='#FFFFFF',
                                        nodelist=open_nodes, ax=ax, node_size=nodeSize)
    draw_nodes.set_edgecolor("b")
    open_nodes = [node for node, attribute in graph.nodes(data=True) if attribute['status'] == "closed"]
    draw_nodes = nx.draw_networkx_nodes(graph, pos, node_color='#0000FF',
                                        nodelist=open_nodes, ax=ax, node_size=nodeSize)
    #nx.draw_networkx_nodes(graph, pos,  cmap=plt.cm.Blues, ax = ax, node_size=nodeSize)
    nx.draw_networkx_edges(graph, pos,
                           edge_color='b',
                           width=3.0
                           )

    collChecker.drawObstacles(ax)

    # draw nodes based on solution path
    Gsp = nx.subgraph(graph, solution)
    nx.draw_networkx_nodes(Gsp, pos,
                           node_size=nodeSize,
                           node_color='g')

    # draw edges based on solution path
    nx.draw_networkx_edges(Gsp, pos, alpha=0.8, edge_color='g', width=10, arrows=True)

    if solution is None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[solution[0]],
                               node_size=300,
                               node_color='#00dd00',  ax=ax)
        nx.draw_networkx_labels(graph, pos, labels={solution[0]: "S"},  ax=ax)

        nx.draw_networkx_nodes(graph, pos, nodelist=[solution[-1]],
                               node_size=300,
                               node_color='#DD0000',  ax=ax)
        nx.draw_networkx_labels(graph, pos, labels={solution[-1]: "G"},  ax=ax)
