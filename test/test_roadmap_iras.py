from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.path import SHPath
import semantic_hierarchical_graph.visualization as vis
import semantic_hierarchical_graph.utils as util
from semantic_hierarchical_graph.types.position import Position


def main():

    G = SHGraph(root_name="Corridor on floor F2", root_pos=Position(0, 0, 0))

    # floor_f2.add_child_by_name(name="Elevator", pos=Position(-12, -10, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Toilets", pos=Position(-6, -8, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Corridor", pos=Position(0, -4, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Seminar Room 1", pos=Position(-4, -2, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Seminar Room 2", pos=Position(-4, 6, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Soccer", pos=Position(0, -8, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Seminar Room 3", pos=Position(6, -8, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Office", pos=Position(0, 2, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Kitchen", pos=Position(8, 2, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Terrace1", pos=Position(-4, 14, 0), is_leaf=True)
    # floor_f2.add_child_by_name(name="Terrace2", pos=Position(16, -4, 0), is_leaf=True)

    G.add_child_by_name(name="Elevator_(-12, -10)_bridge", pos=Position(-12, -10, 2), is_leaf=True)
    G.add_child_by_name(name="(-12, -10)", pos=Position(-12, -10, 0), is_leaf=True)

    # Corridor layout
    G.add_child_by_name(name="(-12, -8)", pos=Position(-12, -8, 0), is_leaf=True)
    G.add_child_by_name(name="(-2, -8)", pos=Position(-2, -8, 0), is_leaf=True)
    G.add_child_by_name(name="(-2, -2)", pos=Position(-2, -2, 0), is_leaf=True)
    G.add_child_by_name(name="(-2, 0)", pos=Position(-2, 0, 0), is_leaf=True)
    G.add_child_by_name(name="(-2, 6)", pos=Position(-2, 6, 0), is_leaf=True)
    G.add_child_by_name(name="(0, 0)", pos=Position(0, 0, 0), is_leaf=True)
    G.add_child_by_name(name="(6, 0)", pos=Position(6, 0, 0), is_leaf=True)
    G.add_child_by_name(name="(8, 0)", pos=Position(8, 0, 0), is_leaf=True)

    G.add_child_by_name(name="Seminar Room 1_(-4, -2)_bridge", pos=Position(-4, -2, 2), is_leaf=True)
    G.add_child_by_name(name="(-4, -2)", pos=Position(-4, -2, 0), is_leaf=True)
    G.add_child_by_name(name="Seminar Room 2_(-4, 6)_bridge", pos=Position(-4, 6, 2), is_leaf=True)
    G.add_child_by_name(name="(-4, 6)", pos=Position(-4, 6, 0), is_leaf=True)
    G.add_child_by_name(name="Soccer_(0, -6)_bridge", pos=Position(0, -6, 2), is_leaf=True)
    G.add_child_by_name(name="(0, -6)", pos=Position(0, -6, 0), is_leaf=True)
    G.add_child_by_name(name="Seminar Room 3_(6, -8)_bridge", pos=Position(6, -8, 2), is_leaf=True)
    G.add_child_by_name(name="(6, -8)", pos=Position(6, -8, 0), is_leaf=True)
    G.add_child_by_name(name="Office_(0, 2)_bridge", pos=Position(0, 2, 2), is_leaf=True)
    G.add_child_by_name(name="(0, 2)", pos=Position(0, 2, 0), is_leaf=True)
    G.add_child_by_name(name="Kitchen_(8, 2)_bridge", pos=Position(8, 2, 2), is_leaf=True)
    G.add_child_by_name(name="(8, 2)", pos=Position(8, 2, 0), is_leaf=True)
    G.add_child_by_name(name="Terrace1_(-2, 12)_bridge", pos=Position(-2, 12, 2), is_leaf=True)
    G.add_child_by_name(name="(-2, 12)", pos=Position(-2, 12, 0), is_leaf=True)
    G.add_child_by_name(name="Terrace2_(12, 0)_bridge", pos=Position(12, 0, 2), is_leaf=True)
    G.add_child_by_name(name="(12, 0)", pos=Position(12, 0, 0), is_leaf=True)

    G.add_connection_recursive(["Elevator_(-12, -10)_bridge"], ["(-12, -10)"])
    G.add_connection_recursive(["(-12, -10)"], ["(-12, -8)"])
    G.add_connection_recursive(["(-12, -8)"], ["(-2, -8)"])
    G.add_connection_recursive(["(-2, -8)"], ["(-2, -2)"])
    G.add_connection_recursive(["(-2, -2)"], ["(-2, 0)"])
    G.add_connection_recursive(["(-2, 0)"], ["(-2, 6)"])
    G.add_connection_recursive(["(-2, -2)"], ["(-4, -2)"])
    G.add_connection_recursive(["(-4, -2)"], ["Seminar Room 1_(-4, -2)_bridge"])
    G.add_connection_recursive(["(-2, 6)"], ["(-4, 6)"])
    G.add_connection_recursive(["(-4, 6)"], ["Seminar Room 2_(-4, 6)_bridge"])
    G.add_connection_recursive(["(0, 0)"], ["(0, -6)"])
    G.add_connection_recursive(["(0, -6)"], ["Soccer_(0, -6)_bridge"])
    G.add_connection_recursive(["(-2, 6)"], ["(-2, 12)"])
    G.add_connection_recursive(["(-2, 12)"], ["Terrace1_(-2, 12)_bridge"])
    G.add_connection_recursive(["(-2, 0)"], ["(0, 0)"])
    G.add_connection_recursive(["(0, 0)"], ["(0, 2)"])
    G.add_connection_recursive(["(0, 2)"], ["Office_(0, 2)_bridge"])
    G.add_connection_recursive(["(0, 0)"], ["(6, 0)"])
    G.add_connection_recursive(["(6, 0)"], ["(6, -8)"])
    G.add_connection_recursive(["(6, -8)"], ["Seminar Room 3_(6, -8)_bridge"])
    G.add_connection_recursive(["(6, 0)"], ["(8, 0)"])
    G.add_connection_recursive(["(8, 0)"], ["(8, 2)"])
    G.add_connection_recursive(["(8, 2)"], ["Kitchen_(8, 2)_bridge"])
    G.add_connection_recursive(["(8, 0)"], ["(12, 0)"])
    G.add_connection_recursive(["(12, 0)"], ["Terrace2_(12, 0)_bridge"])



    path_dict, distance = G.plan_recursive(["Terrace2_(12, 0)_bridge"], ["Elevator_(-12, -10)_bridge"])

    vis.draw_child_graph_3d(G, path_dict, is_leaf=True)

    return G


if __name__ == "__main__":
    main()