from semantic_hierarchical_graph.graph import SHGraph
import semantic_hierarchical_graph.visualization as vis
import semantic_hierarchical_graph.utils as util
from semantic_hierarchical_graph.types.position import Position


def main():
    # G = nx.Graph(name="LTC Campus", level=0, pos=Position(0, 0, 0))
    # child = nx.Graph()
    # G.add_node(child, name="Building F", level=1, parent=G, pos=Position(1, 1, 0))
    # G.add_node(nx.Graph(), name="Building A", level=1, parent=G, pos=Position(1, 0, 0))
    # print(G.nodes[child])
    # print([data["name"] for n, data in G.nodes(data=True)])

    G = SHGraph(root_name="LTC Campus", root_pos=Position(0, 0, 0))

    # build_F = G.add_child_by_name(name="Building F", pos=Position(-5, 0, 0))
    # floor_0 = build_F.add_child_by_name(name="Floor 0", pos=Position(0, 0, 0))
    # floor_0.add_child_by_name(name="Staircase", pos=Position(0, -1, 0))
    # G.add_connection("Building F", "Building A", name="path_1")

    build_a = G.add_child_by_hierarchy(hierarchy=[], name="Building A", pos=Position(0, 0, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building B", pos=Position(5, 0, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building C", pos=Position(5, 5, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building D", pos=Position(0, 5, 0))
    G.add_child_by_hierarchy(hierarchy=[], name="Building E", pos=Position(-5, 5, 0))
    build_f = G.add_child_by_hierarchy(hierarchy=[], name="Building F", pos=Position(-5, 0, 0))
    floor_f0 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 0", pos=Position(0, 0, 0))
    floor_f1 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 1", pos=Position(0, 0, 1))
    floor_f2 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 2", pos=Position(0, 0, 2))
    floor_f3 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 3", pos=Position(0, 0, 3))
    floor_a0 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 0", pos=Position(0, 0, 0))
    floor_a1 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 1", pos=Position(0, 0, 1))
    floor_a2 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 2", pos=Position(0, 0, 2))
    floor_a3 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 3", pos=Position(0, 0, 3))
    floor_f0.add_child_by_name(name="Staircase", pos=Position(0, -1, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Lab", pos=Position(0, 0, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Workshop", pos=Position(1, -1, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="RoboEduLab", pos=Position(0, 1, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Staircase", pos=Position(0, -1, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="IRAS", pos=Position(0, 1, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="xLab", pos=Position(1, 1, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Kitchen", pos=Position(2, 1, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Corridor", pos=Position(1, 0, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Meeting Room", pos=Position(1, -1, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Staircase", pos=Position(0, -1, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Corridor", pos=Position(1, 0, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Seminar Room 1", pos=Position(0, 0, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Seminar Room 2", pos=Position(0, 1, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Seminar Room 3", pos=Position(2, -1, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Office", pos=Position(2, 1, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Kitchen", pos=Position(3, 1, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Staircase", pos=Position(0, -1, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Office", pos=Position(1, 0, 0), is_leaf=True)
    floor_a0.add_child_by_name(name="Staircase", pos=Position(0, -1, 0), is_leaf=True)
    floor_a0.add_child_by_name(name="Entrance", pos=Position(1, 0, 0), is_leaf=True)
    floor_a1.add_child_by_name(name="Cantina", pos=Position(1, 0, 0), is_leaf=True)
    floor_a1.add_child_by_name(name="Staircase", pos=Position(0, -1, 0), is_leaf=True)

    # print(G.get_dict())
    # print(G.get_childs("name"))
    # [print(node.pos, node.pos_abs) for node in G._get_child("Building F").get_childs()]

    G.add_connection_recursive(["Building F", "Floor 0", "Staircase"],
                               ["Building F", "Floor 0", "Lab"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building F", "Floor 0", "Workshop"], name="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building F", "Floor 0", "RoboEduLab"], name="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Staircase"],
                               ["Building F", "Floor 1", "Staircase"], distance=4.0, name="stair_F")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building A", "Floor 0", "Entrance"], name="terrace_door")
    G.add_connection_recursive(["Building F", "Floor 1", "Staircase"],
                               ["Building F", "Floor 1", "Corridor"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "IRAS"], name="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "xLab"], name="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Meeting Room"], name="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Kitchen"], name="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Staircase"],
                               ["Building F", "Floor 2", "Staircase"], distance=4.0, name="stair_F")
    G.add_connection_recursive(["Building F", "Floor 2", "Staircase"],
                               ["Building F", "Floor 2", "Corridor"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 1"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 2"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 3"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Office"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Kitchen"], name="open")
    G.add_connection_recursive(["Building F", "Floor 2", "Staircase"],
                               ["Building F", "Floor 3", "Staircase"], distance=4.0, name="stair_F")
    G.add_connection_recursive(["Building F", "Floor 3", "Staircase"],
                               ["Building F", "Floor 3", "Office"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 1", "Kitchen"],
                               ["Building A", "Floor 1", "Cantina"], distance=10.0, name="terrace_door")
    G.add_connection_recursive(["Building A", "Floor 1", "Staircase"],
                               ["Building A", "Floor 1", "Cantina"], name="floor_door")
    G.add_connection_recursive(["Building A", "Floor 1", "Staircase"],
                               ["Building A", "Floor 0", "Staircase"], distance=4.0, name="stair_A")
    G.add_connection_recursive(["Building A", "Floor 0", "Staircase"],
                               ["Building A", "Floor 0", "Entrance"], name="floor_door")

    # G.save_graph("data/graph.pickle")
    # G = SHGraph.load_graph("data/graph.pickle")

    path = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 1", "Cantina"])
    # path = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building F", "Floor 3", "Office"])
    # path = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])

    # leaf_path_list = G.plan_in_leaf_graph(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])

    path_dict = path.to_dict()
    path.to_json("data/path.json")

    print(util.path_to_list(path_dict, [], with_names=True, is_leaf=True))

    vis.draw_child_graph(G, path_dict)
    vis.draw_child_graph(build_f, path_dict, view_axis="x")
    vis.draw_child_graph(build_a, path_dict, view_axis="x")
    vis.draw_child_graph(floor_f0, path_dict)
    vis.draw_child_graph(floor_f1, path_dict)
    vis.draw_child_graph(floor_a1, path_dict)
    vis.draw_child_graph(floor_a0, path_dict)

    vis.draw_child_graph_3d(floor_f1, path_dict)
    vis.draw_child_graph_3d(G, path_dict, is_leaf=True)

    return G


if __name__ == "__main__":
    main()
