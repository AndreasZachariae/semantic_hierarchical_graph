from semantic_hierarchical_graph.graph import SHGraph
import semantic_hierarchical_graph.visualization as vis
import semantic_hierarchical_graph.utils as util


def main():
    # G = nx.Graph(name="LTC Campus", level=0, pos=(0, 0, 0))
    # child = nx.Graph()
    # G.add_node(child, name="Building F", level=1, parent=G, pos=(1, 1, 0))
    # G.add_node(nx.Graph(), name="Building A", level=1, parent=G, pos=(1, 0, 0))
    # print(G.nodes[child])
    # print([data["name"] for n, data in G.nodes(data=True)])

    G = SHGraph(root_name="LTC Campus", root_pos=(0, 0, 0))

    # build_F = G.add_child_by_name(name="Building F", pos=(-5, 0, 0))
    # floor_0 = build_F.add_child_by_name(name="Floor 0", pos=(0, 0, 0))
    # floor_0.add_child_by_name(name="Staircase", pos=(0, -1, 0))
    # G.add_connection("Building F", "Building A", name="path_1", type="path")

    G.add_child_by_hierarchy(hierarchy=[], name="Building A", pos=(0, 0, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building B", pos=(5, 0, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building C", pos=(5, 5, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building D", pos=(0, 5, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building E", pos=(-5, 5, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building F", pos=(-5, 0, 0))
    G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 0", pos=(0, 0, 0))
    G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 1", pos=(0, 0, 1))
    G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 2", pos=(0, 0, 2))
    G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 3", pos=(0, 0, 3))
    G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 0", pos=(0, 0, 0))
    G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 1", pos=(0, 0, 1))
    G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 2", pos=(0, 0, 2))
    G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 3", pos=(0, 0, 3))
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 1"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 1"], name="IRAS", pos=(0, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 1"], name="xLab", pos=(1, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 1"], name="Kitchen", pos=(2, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 1"], name="Corridor", pos=(1, 0, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 1"], name="Meeting Room", pos=(1, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 0"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 0"], name="Lab", pos=(0, 0, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 0"], name="Workshop", pos=(1, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 0"], name="RoboEduLab", pos=(0, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Corridor", pos=(1, 0, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Seminar Room 1", pos=(0, 0, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Seminar Room 2", pos=(0, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Seminar Room 3", pos=(2, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Office", pos=(2, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 2"], name="Kitchen", pos=(3, 1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 3"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building F", "Floor 3"], name="Office", pos=(1, 0, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building A", "Floor 1"], name="Cantina", pos=(1, 0, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building A", "Floor 1"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building A", "Floor 0"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child_by_hierarchy(hierarchy=["Building A", "Floor 0"], name="Entrance", pos=(1, 0, 0), is_leaf=True)

    # print(G.get_dict())
    # print(G.get_childs("name"))
    # [print(node.pos, node.pos_abs) for node in G._get_child("Building F").get_childs()]

    G.add_connection_recursive(["Building F", "Floor 0", "Staircase"],
                               ["Building F", "Floor 0", "Lab"], name="floor_door", type="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building F", "Floor 0", "Workshop"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building F", "Floor 0", "RoboEduLab"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Staircase"],
                               ["Building F", "Floor 1", "Staircase"], distance=4.0, name="stair_F", type="stair")
    G.add_connection_recursive(["Building F", "Floor 1", "Staircase"],
                               ["Building F", "Floor 1", "Corridor"], name="floor_door", type="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "IRAS"], name="open", type="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "xLab"], name="open", type="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Meeting Room"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Kitchen"], name="open", type="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Staircase"],
                               ["Building F", "Floor 2", "Staircase"], distance=4.0, name="stair_F", type="stair")
    G.add_connection_recursive(["Building F", "Floor 2", "Staircase"],
                               ["Building F", "Floor 2", "Corridor"], name="floor_door", type="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 1"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 2"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 3"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Office"], name="door", type="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Kitchen"], name="open", type="open")
    G.add_connection_recursive(["Building F", "Floor 2", "Staircase"],
                               ["Building F", "Floor 3", "Staircase"], distance=4.0, name="stair_F", type="stair")
    G.add_connection_recursive(["Building F", "Floor 3", "Staircase"],
                               ["Building F", "Floor 3", "Office"], name="floor_door", type="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Kitchen"],
                               ["Building A", "Floor 1", "Cantina"], distance=10.0, name="terrace_door", type="door")
    G.add_connection_recursive(["Building A", "Floor 1", "Staircase"],
                               ["Building A", "Floor 1", "Cantina"], name="floor_door", type="door")
    G.add_connection_recursive(["Building A", "Floor 1", "Staircase"],
                               ["Building A", "Floor 0", "Staircase"], distance=4.0, name="stair_A", type="stair")
    G.add_connection_recursive(["Building A", "Floor 0", "Staircase"],
                               ["Building A", "Floor 0", "Entrance"], name="floor_door", type="door")

    path_dict = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 1", "Cantina"])
    # path_dict = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building F", "Floor 3", "Office"])
    # path_dict = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])

    # leaf_path_list = G.plan_in_leaf_graph(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])

    # util.save_dict_to_json(path_dict, "data/path.json")

    leaf_path_list = util.path_dict_to_leaf_path_list(path_dict)
    print(util.map_names_to_nodes(leaf_path_list))

    # child_path_list = util.path_dict_to_child_path_list(path_dict, [])
    # print(util.map_names_to_nodes(child_path_list))

    # vis.draw_child_graph(G, [], path_dict)
    # vis.draw_child_graph(G, ["Building F"], path_dict, view_axis=0)
    # vis.draw_child_graph(G, ["Building A"], path_dict, view_axis=0)
    # vis.draw_child_graph(G, ["Building F", "Floor 0"], path_dict)
    # vis.draw_child_graph(G, ["Building F", "Floor 2"], path_dict)
    # vis.draw_child_graph(G, ["Building A", "Floor 1"], path_dict)
    # vis.draw_child_graph(G, ["Building A", "Floor 0"], path_dict)

    vis.draw_graph_3d(G.leaf_graph, leaf_path_list)

    return G


if __name__ == "__main__":
    main()
