from semantic_hierarchical_graph.graph import SHGraph
from semantic_hierarchical_graph.path import SHPath
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
    # floor_0.add_child_by_name(name="Elevator", pos=Position(0, -1, 0))
    # G.add_connection("Building F", "Building A", name="path_1")

    # Buildings
    build_a = G.add_child_by_hierarchy(hierarchy=[], name="Building A", pos=Position(50, 12, 0))
    build_b = G.add_child_by_hierarchy(hierarchy=[], name="Building B", pos=Position(86, 12, 0))
    build_c = G.add_child_by_hierarchy(hierarchy=[], name="Building C", pos=Position(86, 48, 0))
    build_d = G.add_child_by_hierarchy(hierarchy=[], name="Building D", pos=Position(50, 48, 0))
    build_e = G.add_child_by_hierarchy(hierarchy=[], name="Building E", pos=Position(14, 48, 0))
    build_f = G.add_child_by_hierarchy(hierarchy=[], name="Building F", pos=Position(14, 12, 0))
    build_t = G.add_child_by_hierarchy(hierarchy=[], name="Terrace", pos=Position(50, 30, 0))

    # Floors
    floor_au = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor -1", pos=Position(0, 0, -4))
    floor_a0 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 0", pos=Position(0, 0, 0))
    floor_a1 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 1", pos=Position(0, 0, 4))
    floor_a2 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 2", pos=Position(0, 0, 8))
    floor_a3 = G.add_child_by_hierarchy(hierarchy=["Building A"], name="Floor 3", pos=Position(0, 0, 12))

    floor_bu = G.add_child_by_hierarchy(hierarchy=["Building B"], name="Floor -1", pos=Position(0, 0, -4))
    floor_b0 = G.add_child_by_hierarchy(hierarchy=["Building B"], name="Floor 0", pos=Position(0, 0, 0))
    floor_b1 = G.add_child_by_hierarchy(hierarchy=["Building B"], name="Floor 1", pos=Position(0, 0, 4))
    floor_b2 = G.add_child_by_hierarchy(hierarchy=["Building B"], name="Floor 2", pos=Position(0, 0, 8))
    floor_b3 = G.add_child_by_hierarchy(hierarchy=["Building B"], name="Floor 3", pos=Position(0, 0, 12))

    floor_cu = G.add_child_by_hierarchy(hierarchy=["Building C"], name="Floor -1", pos=Position(0, 0, -4))
    floor_c0 = G.add_child_by_hierarchy(hierarchy=["Building C"], name="Floor 0", pos=Position(0, 0, 0))
    floor_c1 = G.add_child_by_hierarchy(hierarchy=["Building C"], name="Floor 1", pos=Position(0, 0, 4))
    floor_c2 = G.add_child_by_hierarchy(hierarchy=["Building C"], name="Floor 2", pos=Position(0, 0, 8))
    floor_c3 = G.add_child_by_hierarchy(hierarchy=["Building C"], name="Floor 3", pos=Position(0, 0, 12))

    floor_d0 = G.add_child_by_hierarchy(hierarchy=["Building D"], name="Floor 0", pos=Position(0, 0, 0))
    floor_d1 = G.add_child_by_hierarchy(hierarchy=["Building D"], name="Floor 1", pos=Position(0, 0, 4))
    floor_d2 = G.add_child_by_hierarchy(hierarchy=["Building D"], name="Floor 2", pos=Position(0, 0, 8))

    floor_eu = G.add_child_by_hierarchy(hierarchy=["Building E"], name="Floor -1", pos=Position(0, 0, -4))
    floor_e0 = G.add_child_by_hierarchy(hierarchy=["Building E"], name="Floor 0", pos=Position(0, 0, 0))
    floor_e1 = G.add_child_by_hierarchy(hierarchy=["Building E"], name="Floor 1", pos=Position(0, 0, 4))
    floor_e2 = G.add_child_by_hierarchy(hierarchy=["Building E"], name="Floor 2", pos=Position(0, 0, 8))
    floor_e3 = G.add_child_by_hierarchy(hierarchy=["Building E"], name="Floor 3", pos=Position(0, 0, 12))

    # floor_fu = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor -1", pos=Position(0, 0, -4))
    floor_f0 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 0", pos=Position(0, 0, 0))
    floor_f1 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 1", pos=Position(0, 0, 4))
    floor_f2 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 2", pos=Position(0, 0, 8))
    floor_f3 = G.add_child_by_hierarchy(hierarchy=["Building F"], name="Floor 3", pos=Position(0, 0, 12))

    floor_t0 = G.add_child_by_hierarchy(hierarchy=["Terrace"], name="Floor 0", pos=Position(0, 0, 0))
    floor_t1 = G.add_child_by_hierarchy(hierarchy=["Terrace"], name="Floor 1", pos=Position(0, 0, 4))
    floor_t2 = G.add_child_by_hierarchy(hierarchy=["Terrace"], name="Floor 2", pos=Position(0, 0, 8))
    floor_t3 = G.add_child_by_hierarchy(hierarchy=["Terrace"], name="Floor 3", pos=Position(0, 0, 12))

    # Rooms
    floor_f0.add_child_by_name(name="Elevator", pos=Position(-12, -10, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Toilets", pos=Position(-6, -8, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Lab", pos=Position(0, 0, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Workshop", pos=Position(10, -8, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="RoboEduLab", pos=Position(0, 18, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Terrace", pos=Position(-10, 22, 0), is_leaf=True)
    floor_f0.add_child_by_name(name="Loading Ramp", pos=Position(-14, 22, 0), is_leaf=True)

    floor_f1.add_child_by_name(name="Elevator", pos=Position(-12, -10, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Toilets", pos=Position(-6, -8, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Prof Office", pos=Position(-12, 2, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="IRAS", pos=Position(-6, 2, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Server", pos=Position(0, -8, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="xLab", pos=Position(0, 2, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Booth1", pos=Position(-2, 0, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Booth2", pos=Position(-2, 0, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Kitchen", pos=Position(16, 6, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Corridor", pos=Position(0, -4, 0), is_leaf=True)
    floor_f1.add_child_by_name(name="Meeting Room", pos=Position(6, -8, 0), is_leaf=True)

    floor_f2.add_child_by_name(name="Elevator", pos=Position(-12, -10, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Toilets", pos=Position(-6, -8, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Corridor", pos=Position(0, -4, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Seminar Room 1", pos=Position(-4, -2, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Seminar Room 2", pos=Position(-4, 6, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Soccer", pos=Position(0, -8, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Seminar Room 3", pos=Position(6, -8, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Office", pos=Position(0, 2, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Kitchen", pos=Position(8, 2, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Terrace1", pos=Position(-4, 14, 0), is_leaf=True)
    floor_f2.add_child_by_name(name="Terrace2", pos=Position(16, -4, 0), is_leaf=True)

    floor_f3.add_child_by_name(name="Elevator", pos=Position(-12, -10, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Toilets", pos=Position(-6, -8, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Corridor", pos=Position(0, 0, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Kitchen", pos=Position(-4, -4, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Meeting Room", pos=Position(-12, 2, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Office1", pos=Position(0, 2, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Office2", pos=Position(4, -8, 0), is_leaf=True)
    floor_f3.add_child_by_name(name="Terrace", pos=Position(8, 0, 0), is_leaf=True)

    G.add_connection_recursive(["Building F", "Floor 0", "Elevator"],
                               ["Building F", "Floor 1", "Elevator"], distance=4.0, name="stair_F")
    G.add_connection_recursive(["Building F", "Floor 0", "Elevator"],
                               ["Building F", "Floor 0", "Lab"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 0", "Elevator"],
                               ["Building F", "Floor 0", "Toilets"], name="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building F", "Floor 0", "Workshop"], name="door")
    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Building F", "Floor 0", "RoboEduLab"], name="door")
    G.add_connection_recursive(["Building F", "Floor 0", "RoboEduLab"],
                               ["Building F", "Floor 0", "Terrace"], name="outside_door")
    G.add_connection_recursive(["Building F", "Floor 0", "Terrace"],
                               ["Building F", "Floor 0", "Loading Ramp"], name="open")

    G.add_connection_recursive(["Building F", "Floor 1", "Elevator"],
                               ["Building F", "Floor 2", "Elevator"], distance=4.0, name="stair_F")
    G.add_connection_recursive(["Building F", "Floor 1", "Elevator"],
                               ["Building F", "Floor 1", "Corridor"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 1", "Elevator"],
                               ["Building F", "Floor 1", "Toilets"], name="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "IRAS"], name="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "xLab"], name="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Meeting Room"], name="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Kitchen"], name="open")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Prof Office"], name="door")
    G.add_connection_recursive(["Building F", "Floor 1", "Corridor"],
                               ["Building F", "Floor 1", "Server"], name="door")
    G.add_connection_recursive(["Building F", "Floor 1", "xLab"],
                               ["Building F", "Floor 1", "Booth1"], name="door")
    G.add_connection_recursive(["Building F", "Floor 1", "xLab"],
                               ["Building F", "Floor 1", "Booth2"], name="door")

    G.add_connection_recursive(["Building F", "Floor 2", "Elevator"],
                               ["Building F", "Floor 3", "Elevator"], distance=4.0, name="stair_F")
    G.add_connection_recursive(["Building F", "Floor 2", "Elevator"],
                               ["Building F", "Floor 2", "Corridor"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 2", "Elevator"],
                               ["Building F", "Floor 2", "Toilets"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 1"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 2"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Seminar Room 3"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Office"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Soccer"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Kitchen"], name="open")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Terrace1"], name="door")
    G.add_connection_recursive(["Building F", "Floor 2", "Corridor"],
                               ["Building F", "Floor 2", "Terrace2"], name="outside_door")

    G.add_connection_recursive(["Building F", "Floor 3", "Elevator"],
                               ["Building F", "Floor 3", "Corridor"], name="floor_door")
    G.add_connection_recursive(["Building F", "Floor 3", "Elevator"],
                               ["Building F", "Floor 3", "Toilets"], name="door")
    G.add_connection_recursive(["Building F", "Floor 3", "Kitchen"],
                               ["Building F", "Floor 3", "Corridor"], name="open")
    G.add_connection_recursive(["Building F", "Floor 3", "Meeting Room"],
                               ["Building F", "Floor 3", "Corridor"], name="door")
    G.add_connection_recursive(["Building F", "Floor 3", "Office1"],
                               ["Building F", "Floor 3", "Corridor"], name="open")
    G.add_connection_recursive(["Building F", "Floor 3", "Office2"],
                               ["Building F", "Floor 3", "Corridor"], name="open")
    G.add_connection_recursive(["Building F", "Floor 3", "Terrace"],
                               ["Building F", "Floor 3", "Corridor"], name="outside_door")

    floor_a0.add_child_by_name(name="Elevator", pos=Position(-6, -10, 0), is_leaf=True)
    floor_a0.add_child_by_name(name="Toilets", pos=Position(0, -8, 0), is_leaf=True)
    floor_a0.add_child_by_name(name="Lobby", pos=Position(4, 0, 0), is_leaf=True)
    floor_a0.add_child_by_name(name="Office", pos=Position(4, -8, 0), is_leaf=True)

    floor_a1.add_child_by_name(name="Elevator", pos=Position(-6, -10, 0), is_leaf=True)
    floor_a1.add_child_by_name(name="Toilets", pos=Position(0, -8, 0), is_leaf=True)
    floor_a1.add_child_by_name(name="Cantina", pos=Position(4, 0, 0), is_leaf=True)
    floor_a1.add_child_by_name(name="Kitchen", pos=Position(4, -8, 0), is_leaf=True)

    floor_a2.add_child_by_name(name="Elevator", pos=Position(-6, -10, 0), is_leaf=True)
    floor_a2.add_child_by_name(name="Toilets", pos=Position(0, -8, 0), is_leaf=True)
    floor_a2.add_child_by_name(name="Conference Room", pos=Position(4, 0, 0), is_leaf=True)

    floor_a3.add_child_by_name(name="Elevator", pos=Position(-6, -10, 0), is_leaf=True)
    floor_a3.add_child_by_name(name="Toilets", pos=Position(0, -8, 0), is_leaf=True)
    floor_a3.add_child_by_name(name="Conference Room", pos=Position(4, 0, 0), is_leaf=True)

    G.add_connection_recursive(["Building A", "Floor 1", "Elevator"],
                               ["Building A", "Floor 0", "Elevator"], distance=4.0, name="stair_A")
    G.add_connection_recursive(["Building A", "Floor 0", "Elevator"],
                               ["Building A", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building A", "Floor 0", "Office"],
                               ["Building A", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building A", "Floor 0", "Toilets"],
                               ["Building A", "Floor 0", "Lobby"], name="door")

    G.add_connection_recursive(["Building A", "Floor 1", "Elevator"],
                               ["Building A", "Floor 2", "Elevator"], distance=4.0, name="stair_A")
    G.add_connection_recursive(["Building A", "Floor 1", "Elevator"],
                               ["Building A", "Floor 1", "Cantina"], name="floor_door")
    G.add_connection_recursive(["Building A", "Floor 1", "Cantina"],
                               ["Building A", "Floor 1", "Kitchen"], name="door")
    G.add_connection_recursive(["Building A", "Floor 1", "Toilets"],
                               ["Building A", "Floor 1", "Cantina"], name="door")

    G.add_connection_recursive(["Building A", "Floor 3", "Elevator"],
                               ["Building A", "Floor 2", "Elevator"], distance=4.0, name="stair_A")
    G.add_connection_recursive(["Building A", "Floor 2", "Elevator"],
                               ["Building A", "Floor 2", "Conference Room"], name="floor_door")
    G.add_connection_recursive(["Building A", "Floor 2", "Toilets"],
                               ["Building A", "Floor 2", "Conference Room"], name="door")

    G.add_connection_recursive(["Building A", "Floor 3", "Elevator"],
                               ["Building A", "Floor 3", "Conference Room"], name="floor_door")
    G.add_connection_recursive(["Building A", "Floor 3", "Toilets"],
                               ["Building A", "Floor 3", "Conference Room"], name="door")

    floor_b0.add_child_by_name(name="Elevator", pos=Position(12, -10, 0), is_leaf=True)
    floor_b0.add_child_by_name(name="Toilets", pos=Position(6, -8, 0), is_leaf=True)
    floor_b0.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_b0.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_b0.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)
    floor_b0.add_child_by_name(name="Office2", pos=Position(-12, 8, 0), is_leaf=True)
    floor_b0.add_child_by_name(name="Meeting Room", pos=Position(8, 10, 0), is_leaf=True)

    floor_b1.add_child_by_name(name="Elevator", pos=Position(12, -10, 0), is_leaf=True)
    floor_b1.add_child_by_name(name="Toilets", pos=Position(6, -8, 0), is_leaf=True)
    floor_b1.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_b1.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_b1.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)
    floor_b1.add_child_by_name(name="Office2", pos=Position(-12, 8, 0), is_leaf=True)
    floor_b1.add_child_by_name(name="Meeting Room", pos=Position(8, 10, 0), is_leaf=True)

    floor_b2.add_child_by_name(name="Elevator", pos=Position(12, -10, 0), is_leaf=True)
    floor_b2.add_child_by_name(name="Toilets", pos=Position(6, -8, 0), is_leaf=True)
    floor_b2.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_b2.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_b2.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)
    floor_b2.add_child_by_name(name="Meeting Room", pos=Position(8, 10, 0), is_leaf=True)

    floor_b3.add_child_by_name(name="Elevator", pos=Position(12, -10, 0), is_leaf=True)
    floor_b3.add_child_by_name(name="Toilets", pos=Position(6, -8, 0), is_leaf=True)
    floor_b3.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_b3.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_b3.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)

    G.add_connection_recursive(["Building B", "Floor 1", "Elevator"],
                               ["Building B", "Floor 0", "Elevator"], distance=4.0, name="stair_B")
    G.add_connection_recursive(["Building B", "Floor 0", "Elevator"],
                               ["Building B", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 0", "Toilets"],
                               ["Building B", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building B", "Floor 0", "Corridor"],
                               ["Building B", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 0", "Office1"],
                               ["Building B", "Floor 0", "Corridor"], name="door")
    G.add_connection_recursive(["Building B", "Floor 0", "Office2"],
                               ["Building B", "Floor 0", "Corridor"], name="door")
    G.add_connection_recursive(["Building B", "Floor 0", "Meeting Room"],
                               ["Building B", "Floor 0", "Corridor"], name="door")

    G.add_connection_recursive(["Building B", "Floor 2", "Elevator"],
                               ["Building B", "Floor 1", "Elevator"], distance=4.0, name="stair_B")
    G.add_connection_recursive(["Building B", "Floor 1", "Elevator"],
                               ["Building B", "Floor 1", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 1", "Toilets"],
                               ["Building B", "Floor 1", "Lobby"], name="door")
    G.add_connection_recursive(["Building B", "Floor 1", "Corridor"],
                               ["Building B", "Floor 1", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 1", "Office1"],
                               ["Building B", "Floor 1", "Corridor"], name="door")
    G.add_connection_recursive(["Building B", "Floor 1", "Office2"],
                               ["Building B", "Floor 1", "Corridor"], name="door")
    G.add_connection_recursive(["Building B", "Floor 1", "Meeting Room"],
                               ["Building B", "Floor 1", "Corridor"], name="door")

    G.add_connection_recursive(["Building B", "Floor 3", "Elevator"],
                               ["Building B", "Floor 2", "Elevator"], distance=4.0, name="stair_B")
    G.add_connection_recursive(["Building B", "Floor 2", "Elevator"],
                               ["Building B", "Floor 2", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 2", "Toilets"],
                               ["Building B", "Floor 2", "Lobby"], name="door")
    G.add_connection_recursive(["Building B", "Floor 2", "Corridor"],
                               ["Building B", "Floor 2", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 2", "Office1"],
                               ["Building B", "Floor 2", "Corridor"], name="door")
    G.add_connection_recursive(["Building B", "Floor 2", "Meeting Room"],
                               ["Building B", "Floor 2", "Corridor"], name="door")

    G.add_connection_recursive(["Building B", "Floor 3", "Elevator"],
                               ["Building B", "Floor 3", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 3", "Toilets"],
                               ["Building B", "Floor 3", "Lobby"], name="door")
    G.add_connection_recursive(["Building B", "Floor 3", "Corridor"],
                               ["Building B", "Floor 3", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building B", "Floor 3", "Office1"],
                               ["Building B", "Floor 3", "Corridor"], name="door")

    floor_c0.add_child_by_name(name="Elevator", pos=Position(-2, 10, 0), is_leaf=True)
    floor_c0.add_child_by_name(name="Toilets", pos=Position(2, 8, 0), is_leaf=True)
    floor_c0.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_c0.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_c0.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)
    floor_c0.add_child_by_name(name="Office2", pos=Position(-12, 8, 0), is_leaf=True)
    floor_c0.add_child_by_name(name="Meeting Room", pos=Position(8, 10, 0), is_leaf=True)

    floor_c1.add_child_by_name(name="Elevator", pos=Position(-2, 10, 0), is_leaf=True)
    floor_c1.add_child_by_name(name="Toilets", pos=Position(2, 8, 0), is_leaf=True)
    floor_c1.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_c1.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_c1.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)
    floor_c1.add_child_by_name(name="Office2", pos=Position(-12, 8, 0), is_leaf=True)
    floor_c1.add_child_by_name(name="Meeting Room", pos=Position(8, 10, 0), is_leaf=True)

    floor_c2.add_child_by_name(name="Elevator", pos=Position(-2, 10, 0), is_leaf=True)
    floor_c2.add_child_by_name(name="Toilets", pos=Position(2, 8, 0), is_leaf=True)
    floor_c2.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_c2.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_c2.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)
    floor_c2.add_child_by_name(name="Meeting Room", pos=Position(8, 10, 0), is_leaf=True)

    floor_c3.add_child_by_name(name="Elevator", pos=Position(-2, 10, 0), is_leaf=True)
    floor_c3.add_child_by_name(name="Toilets", pos=Position(2, 8, 0), is_leaf=True)
    floor_c3.add_child_by_name(name="Lobby", pos=Position(6, -2, 0), is_leaf=True)
    floor_c3.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_c3.add_child_by_name(name="Office1", pos=Position(8, 2, 0), is_leaf=True)

    G.add_connection_recursive(["Building C", "Floor 1", "Elevator"],
                               ["Building C", "Floor 0", "Elevator"], distance=4.0, name="stair_C")
    G.add_connection_recursive(["Building C", "Floor 0", "Elevator"],
                               ["Building C", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 0", "Toilets"],
                               ["Building C", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building C", "Floor 0", "Corridor"],
                               ["Building C", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 0", "Office1"],
                               ["Building C", "Floor 0", "Corridor"], name="door")
    G.add_connection_recursive(["Building C", "Floor 0", "Office2"],
                               ["Building C", "Floor 0", "Corridor"], name="door")
    G.add_connection_recursive(["Building C", "Floor 0", "Meeting Room"],
                               ["Building C", "Floor 0", "Corridor"], name="door")

    G.add_connection_recursive(["Building C", "Floor 2", "Elevator"],
                               ["Building C", "Floor 1", "Elevator"], distance=4.0, name="stair_C")
    G.add_connection_recursive(["Building C", "Floor 1", "Elevator"],
                               ["Building C", "Floor 1", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 1", "Toilets"],
                               ["Building C", "Floor 1", "Lobby"], name="door")
    G.add_connection_recursive(["Building C", "Floor 1", "Corridor"],
                               ["Building C", "Floor 1", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 1", "Office1"],
                               ["Building C", "Floor 1", "Corridor"], name="door")
    G.add_connection_recursive(["Building C", "Floor 1", "Office2"],
                               ["Building C", "Floor 1", "Corridor"], name="door")
    G.add_connection_recursive(["Building C", "Floor 1", "Meeting Room"],
                               ["Building C", "Floor 1", "Corridor"], name="door")

    G.add_connection_recursive(["Building C", "Floor 3", "Elevator"],
                               ["Building C", "Floor 2", "Elevator"], distance=4.0, name="stair_C")
    G.add_connection_recursive(["Building C", "Floor 2", "Elevator"],
                               ["Building C", "Floor 2", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 2", "Toilets"],
                               ["Building C", "Floor 2", "Lobby"], name="door")
    G.add_connection_recursive(["Building C", "Floor 2", "Corridor"],
                               ["Building C", "Floor 2", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 2", "Office1"],
                               ["Building C", "Floor 2", "Corridor"], name="door")
    G.add_connection_recursive(["Building C", "Floor 2", "Meeting Room"],
                               ["Building C", "Floor 2", "Corridor"], name="door")

    G.add_connection_recursive(["Building C", "Floor 3", "Elevator"],
                               ["Building C", "Floor 3", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 3", "Toilets"],
                               ["Building C", "Floor 3", "Lobby"], name="door")
    G.add_connection_recursive(["Building C", "Floor 3", "Corridor"],
                               ["Building C", "Floor 3", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building C", "Floor 3", "Office1"],
                               ["Building C", "Floor 3", "Corridor"], name="door")

    floor_d0.add_child_by_name(name="Toilets", pos=Position(0, 2, 0), is_leaf=True)
    floor_d0.add_child_by_name(name="Lobby", pos=Position(0, 0, 0), is_leaf=True)
    floor_d0.add_child_by_name(name="Office", pos=Position(-8, 4, 0), is_leaf=True)
    floor_d0.add_child_by_name(name="Fitness", pos=Position(0, 4, 0), is_leaf=True)
    floor_d0.add_child_by_name(name="Terrace", pos=Position(0, 10, 0), is_leaf=True)

    floor_d1.add_child_by_name(name="Toilets", pos=Position(0, 2, 0), is_leaf=True)
    floor_d1.add_child_by_name(name="Fitness", pos=Position(0, 4, 0), is_leaf=True)
    floor_d1.add_child_by_name(name="Terrace", pos=Position(0, 8, 0), is_leaf=True)

    G.add_connection_recursive(["Building D", "Floor 0", "Toilets"],
                               ["Building D", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building D", "Floor 0", "Office"],
                               ["Building D", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building D", "Floor 0", "Fitness"],
                               ["Building D", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building D", "Floor 0", "Fitness"],
                               ["Building D", "Floor 0", "Terrace"], name="door")

    G.add_connection_recursive(["Building D", "Floor 1", "Toilets"],
                               ["Building D", "Floor 1", "Fitness"], name="door")
    G.add_connection_recursive(["Building D", "Floor 1", "Fitness"],
                               ["Building D", "Floor 1", "Terrace"], name="door")

    floor_e0.add_child_by_name(name="Elevator", pos=Position(2, 10, 0), is_leaf=True)
    floor_e0.add_child_by_name(name="Toilets", pos=Position(-2, 8, 0), is_leaf=True)
    floor_e0.add_child_by_name(name="Lobby", pos=Position(-6, -2, 0), is_leaf=True)
    floor_e0.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_e0.add_child_by_name(name="Office1", pos=Position(-8, 2, 0), is_leaf=True)
    floor_e0.add_child_by_name(name="Office2", pos=Position(12, 8, 0), is_leaf=True)
    floor_e0.add_child_by_name(name="Meeting Room", pos=Position(-8, 10, 0), is_leaf=True)

    floor_e1.add_child_by_name(name="Elevator", pos=Position(2, 10, 0), is_leaf=True)
    floor_e1.add_child_by_name(name="Toilets", pos=Position(-2, 8, 0), is_leaf=True)
    floor_e1.add_child_by_name(name="Lobby", pos=Position(-6, -2, 0), is_leaf=True)
    floor_e1.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_e1.add_child_by_name(name="Office1", pos=Position(-8, 2, 0), is_leaf=True)
    floor_e1.add_child_by_name(name="Office2", pos=Position(12, 8, 0), is_leaf=True)
    floor_e1.add_child_by_name(name="Meeting Room", pos=Position(-8, 10, 0), is_leaf=True)

    floor_e2.add_child_by_name(name="Elevator", pos=Position(2, 10, 0), is_leaf=True)
    floor_e2.add_child_by_name(name="Toilets", pos=Position(-2, 8, 0), is_leaf=True)
    floor_e2.add_child_by_name(name="Lobby", pos=Position(-6, -2, 0), is_leaf=True)
    floor_e2.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_e2.add_child_by_name(name="Office1", pos=Position(-8, 2, 0), is_leaf=True)
    floor_e2.add_child_by_name(name="Meeting Room", pos=Position(-8, 10, 0), is_leaf=True)

    floor_e3.add_child_by_name(name="Elevator", pos=Position(2, 10, 0), is_leaf=True)
    floor_e3.add_child_by_name(name="Toilets", pos=Position(-2, 8, 0), is_leaf=True)
    floor_e3.add_child_by_name(name="Lobby", pos=Position(-6, -2, 0), is_leaf=True)
    floor_e3.add_child_by_name(name="Corridor", pos=Position(0, 6, 0), is_leaf=True)
    floor_e3.add_child_by_name(name="Office1", pos=Position(-8, 2, 0), is_leaf=True)

    G.add_connection_recursive(["Building E", "Floor 1", "Elevator"],
                               ["Building E", "Floor 0", "Elevator"], distance=4.0, name="stair_C")
    G.add_connection_recursive(["Building E", "Floor 0", "Elevator"],
                               ["Building E", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 0", "Toilets"],
                               ["Building E", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Building E", "Floor 0", "Corridor"],
                               ["Building E", "Floor 0", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 0", "Office1"],
                               ["Building E", "Floor 0", "Corridor"], name="door")
    G.add_connection_recursive(["Building E", "Floor 0", "Office2"],
                               ["Building E", "Floor 0", "Corridor"], name="door")
    G.add_connection_recursive(["Building E", "Floor 0", "Meeting Room"],
                               ["Building E", "Floor 0", "Corridor"], name="door")

    G.add_connection_recursive(["Building E", "Floor 2", "Elevator"],
                               ["Building E", "Floor 1", "Elevator"], distance=4.0, name="stair_C")
    G.add_connection_recursive(["Building E", "Floor 1", "Elevator"],
                               ["Building E", "Floor 1", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 1", "Toilets"],
                               ["Building E", "Floor 1", "Lobby"], name="door")
    G.add_connection_recursive(["Building E", "Floor 1", "Corridor"],
                               ["Building E", "Floor 1", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 1", "Office1"],
                               ["Building E", "Floor 1", "Corridor"], name="door")
    G.add_connection_recursive(["Building E", "Floor 1", "Office2"],
                               ["Building E", "Floor 1", "Corridor"], name="door")
    G.add_connection_recursive(["Building E", "Floor 1", "Meeting Room"],
                               ["Building E", "Floor 1", "Corridor"], name="door")

    G.add_connection_recursive(["Building E", "Floor 3", "Elevator"],
                               ["Building E", "Floor 2", "Elevator"], distance=4.0, name="stair_C")
    G.add_connection_recursive(["Building E", "Floor 2", "Elevator"],
                               ["Building E", "Floor 2", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 2", "Toilets"],
                               ["Building E", "Floor 2", "Lobby"], name="door")
    G.add_connection_recursive(["Building E", "Floor 2", "Corridor"],
                               ["Building E", "Floor 2", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 2", "Office1"],
                               ["Building E", "Floor 2", "Corridor"], name="door")
    G.add_connection_recursive(["Building E", "Floor 2", "Meeting Room"],
                               ["Building E", "Floor 2", "Corridor"], name="door")

    G.add_connection_recursive(["Building E", "Floor 3", "Elevator"],
                               ["Building E", "Floor 3", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 3", "Toilets"],
                               ["Building E", "Floor 3", "Lobby"], name="door")
    G.add_connection_recursive(["Building E", "Floor 3", "Corridor"],
                               ["Building E", "Floor 3", "Lobby"], name="floor_door")
    G.add_connection_recursive(["Building E", "Floor 3", "Office1"],
                               ["Building E", "Floor 3", "Corridor"], name="door")

    floor_t0.add_child_by_name(name="Elevator", pos=Position(4, -4, 0), is_leaf=True)
    floor_t0.add_child_by_name(name="Garden", pos=Position(0, 0, 0), is_leaf=True)
    floor_t0.add_child_by_name(name="Entrance Stairs", pos=Position(22, -20, 0), is_leaf=True)

    floor_t1.add_child_by_name(name="Elevator", pos=Position(4, -4, 0), is_leaf=True)
    floor_t1.add_child_by_name(name="Ring A", pos=Position(-6, -12, 0), is_leaf=True)
    floor_t1.add_child_by_name(name="Ring B", pos=Position(12, 0, 0), is_leaf=True)
    floor_t1.add_child_by_name(name="Ring D", pos=Position(0, 12, 0), is_leaf=True)
    floor_t1.add_child_by_name(name="Ring F", pos=Position(-12, 0, 0), is_leaf=True)
    floor_t1.add_child_by_name(name="Terrace E", pos=Position(-32, 4, 0), is_leaf=True)
    floor_t1.add_child_by_name(name="Terrace C", pos=Position(36, 2, 0), is_leaf=True)

    floor_t2.add_child_by_name(name="Elevator", pos=Position(4, -4, 0), is_leaf=True)

    floor_t3.add_child_by_name(name="Elevator", pos=Position(4, -4, 0), is_leaf=True)

    G.add_connection_recursive(["Building F", "Floor 0", "Lab"],
                               ["Terrace", "Floor 0", "Garden"], name="outside_door")
    G.add_connection_recursive(["Building C", "Floor 0", "Office2"],
                               ["Terrace", "Floor 0", "Garden"], name="outside_door")
    G.add_connection_recursive(["Building E", "Floor 0", "Office2"],
                               ["Terrace", "Floor 0", "Garden"], name="outside_door")
    G.add_connection_recursive(["Building B", "Floor 0", "Office2"],
                               ["Terrace", "Floor 0", "Garden"], name="outside_door")
    G.add_connection_recursive(["Building D", "Floor 0", "Lobby"],
                               ["Terrace", "Floor 0", "Garden"], name="outside_door")
    G.add_connection_recursive(["Building C", "Floor 1", "Corridor"],
                               ["Terrace", "Floor 1", "Terrace C"], name="outside_door")
    G.add_connection_recursive(["Building E", "Floor 1", "Corridor"],
                               ["Terrace", "Floor 1", "Terrace E"], name="outside_door")
    G.add_connection_recursive(["Building C", "Floor 1", "Corridor"],
                               ["Terrace", "Floor 1", "Terrace C"], name="outside_door")
    G.add_connection_recursive(["Building F", "Floor 1", "Kitchen"],
                               ["Terrace", "Floor 1", "Ring F"], name="outside_door")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring A"],
                               ["Building A", "Floor 1", "Cantina"], name="outside_door")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring B"],
                               ["Building A", "Floor 1", "Cantina"], name="outside_door")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring D"],
                               ["Building D", "Floor 1", "Fitness"], name="outside_door")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring F"],
                               ["Building F", "Floor 2", "Terrace2"], name="stair")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring A"],
                               ["Terrace", "Floor 1", "Ring F"], name="open")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring D"],
                               ["Terrace", "Floor 1", "Ring F"], name="open")
    G.add_connection_recursive(["Terrace", "Floor 1", "Ring D"],
                               ["Terrace", "Floor 1", "Ring B"], name="open")
    G.add_connection_recursive(["Terrace", "Floor 0", "Entrance Stairs"],
                               ["Terrace", "Floor 0", "Garden"], name="open")
    G.add_connection_recursive(["Terrace", "Floor 0", "Entrance Stairs"],
                               ["Building A", "Floor 0", "Lobby"], name="outside_door")
    G.add_connection_recursive(["Terrace", "Floor 1", "Terrace E"],
                               ["Terrace", "Floor 1", "Ring F"], name="open")
    G.add_connection_recursive(["Terrace", "Floor 1", "Terrace C"],
                               ["Terrace", "Floor 1", "Ring B"], name="open")
    G.add_connection_recursive(["Terrace", "Floor 0", "Elevator"],
                               ["Terrace", "Floor 0", "Garden"], name="open")

    G.add_connection_recursive(["Terrace", "Floor 0", "Garden"],
                               ["Terrace", "Floor 1", "Ring D"], distance=15.0, name="open")
    # G.add_connection_recursive(["Terrace", "Floor 0", "Elevator"],
    #                            ["Terrace", "Floor 1", "Elevator"], distance=4.0, name="elevator_T")
    # G.add_connection_recursive(["Terrace", "Floor 1", "Elevator"],
    #                            ["Terrace", "Floor 2", "Elevator"], distance=4.0, name="elevator_T")
    # G.add_connection_recursive(["Terrace", "Floor 2", "Elevator"],
    #                            ["Terrace", "Floor 3", "Elevator"], distance=4.0, name="elevator_T")
    G.add_connection_recursive(["Terrace", "Floor 0", "Elevator"],
                               ["Building A", "Floor 0", "Lobby"], name="door")
    G.add_connection_recursive(["Terrace", "Floor 1", "Elevator"],
                               ["Building A", "Floor 1", "Cantina"], name="door")
    G.add_connection_recursive(["Terrace", "Floor 2", "Elevator"],
                               ["Building A", "Floor 2", "Conference Room"], name="door")
    G.add_connection_recursive(["Terrace", "Floor 3", "Elevator"],
                               ["Building A", "Floor 3", "Conference Room"], name="door")

    # print(G.get_dict())
    # print(G.get_childs("name"))
    # [print(node.pos, node.pos_abs) for node in G._get_child("Building F").get_childs()]

    # G.save_graph("data/graph.pickle")
    # G = SHGraph.load_graph("data/graph.pickle")

    # path_dict, distance = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 1", "Cantina"])
    # path_dict, distance = G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building F", "Floor 3", "Office"])
    path_dict, distance = G.plan_recursive(["Building F", "Floor 3", "Meeting Room"], [
                                           "Building D", "Floor 1", "Fitness"])

    # leaf_path_list = G.plan_in_leaf_graph(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])

    SHPath.save_path(path_dict, "data/path.json")

    print(util.path_to_list(path_dict, [], with_names=True, is_leaf=True))

    # Count all nodes an each nested level by traversing through the whole graph structure:
    nodes_on_l1 = len(G.get_childs())
    nodes_on_l2 = 0
    nodes_on_l3 = 0
    nodes_on_l4 = 20  # Average number of nodes on each roadmap
    nodes_on_l5 = 80*100  # Average number of cells on each gridmap
    for child in G.get_childs():
        nodes_on_l2 += len(child.get_childs())
        for child2 in child.get_childs():
            nodes_on_l3 += len(child2.get_childs())
            for child3 in child2.get_childs():
                nodes_on_l4 += len(child3.get_childs())

    print("Nodes on level 1: ", nodes_on_l1)
    print("Nodes on level 2: ", nodes_on_l2)
    print("Nodes on level 3: ", nodes_on_l3)
    print("Nodes on level 4: ", nodes_on_l4 * nodes_on_l3)
    print("Nodes on level 5: ", nodes_on_l5 * nodes_on_l3)
    print("Total nodes: ", nodes_on_l1 + nodes_on_l2 + nodes_on_l3 +
          nodes_on_l4 * nodes_on_l3)

    vis.draw_child_graph(G, path_dict)
    vis.draw_child_graph(build_f, path_dict, view_axis="x")
    vis.draw_child_graph_3d(build_f, path_dict)
    # vis.draw_child_graph(build_a, path_dict, view_axis="x")
    # vis.draw_child_graph(floor_f0, path_dict)
    vis.draw_child_graph(floor_f2, path_dict)
    # vis.draw_child_graph(floor_a1, path_dict)
    # vis.draw_child_graph(floor_a0, path_dict)

    # vis.draw_child_graph_3d(floor_f1, path_dict)
    vis.draw_child_graph_3d(G, path_dict, is_leaf=True)

    return G


if __name__ == "__main__":
    main()
