import networkx as nx
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, TypeVar, Generic, Optional
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
    'wing': 2,
    'floor': 3,
    'room': 4,
    'location': 5,
    'gridmap': 6
}

# 3 Alternatives:
# 1. One single graph with all levels
# 2. Hierachical graphs
# 3. Dictionary like structure G["building F"]["floor 1"]["room 1"]

T = TypeVar('T', bound='SHNode')


class SHNode(Generic[T]):
    def __init__(self, unique_name: str, parent_node, pos: Tuple[float, float, float], is_root: bool = False, is_leaf: bool = False):
        self.unique_name: str = unique_name
        self.is_root: bool = is_root
        self.is_leaf: bool = is_leaf
        self.pos: Tuple[float, float, float] = pos
        self.pos_abs: Tuple[float, float, float] = tuple(np.add(pos, parent_node.pos_abs)) if not is_root else pos
        self.parent_node: SHNode = parent_node
        self.child_graph: nx.Graph = nx.Graph()

    def _add_child(self, name: str, pos: Tuple[float, float, float], is_leaf: bool = False, **data):
        # Child of type SHNode with unique name on that level

        self.child_graph.add_node(SHNode(unique_name=name, parent_node=self, pos=pos, is_leaf=is_leaf),
                                  name=name, **data)

    def _add_connection(self, child_name_1: str, child_name_2: str, distance: Optional[float] = None, **data):
        child_1 = self._get_child(child_name_1)
        child_2 = self._get_child(child_name_2)
        if distance is None:
            distance = self.get_euclidean_distance(child_1.pos_abs, child_2.pos_abs)
        self.child_graph.add_edge(child_1, child_2, distance=distance, **data)

    def _get_child(self, name: str) -> T:
        # Could be way more efficient if using hierarchy_list as hashable node instead of SHNode object
        # Drawback: node.pos etc is not that easy, has to get the data dict all the time graph.nodes(data=True)
        for node in self.child_graph.nodes:
            if node.unique_name == name:
                return node
        raise ValueError("Child with name {} not found".format(name))

    def get_childs(self, key=None) -> List[T]:
        if key is None:
            return self.child_graph.nodes()  # type: ignore
        else:
            return [value for n, value in self.child_graph.nodes(data=key)]  # type: ignore

    def is_child(self, node: T) -> bool:
        return self.child_graph.has_node(node)

    def get_dict(self) -> dict:
        s = {}
        for node in self.child_graph.nodes:
            s[node.unique_name] = node.get_dict()
        return s

    def get_euclidean_distance(self, pos_1: Tuple, pos_2: Tuple) -> float:
        return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2 + (pos_1[2] - pos_2[2]) ** 2) ** 0.5

    def _plan(self, start_name: str, goal_name: str) -> List[T]:
        return nx.shortest_path(self.child_graph,
                                source=self._get_child(start_name),
                                target=self._get_child(goal_name),
                                weight="distance",
                                method="dijkstra")  # type: ignore

    def _plan_recursive(self, start_hierarchy: List[str], goal_hierarchy: List[str], hierarchy_mask: List[bool],
                        hierarchy_level: int, prev_parent: Optional[T], next_parent: Optional[T]):
        if self.is_leaf:
            print("Leaf reached:", self.unique_name)
            return

        start_name = start_hierarchy[hierarchy_level]
        goal_name = goal_hierarchy[hierarchy_level]

        # if start and goal are not in the same branch
        if hierarchy_mask[hierarchy_level] == False:
            # if start is not on the same branch as parent
            if prev_parent is not None:
                start_name = prev_parent.unique_name + "_h_bridge"
            # if goal/next parent_node is not on same hierarchy as start
            if next_parent is not None:
                goal_name = next_parent.unique_name + "_h_bridge"

        # # if start is not on same hierarchy as parent_node
        # if self.parent_node.unique_name != start_hierarchy[index-1]:
        #     start_name = prev_parent.unique_name + "_h_bridge"

        # # if not the last node in parent_path
        # if i+1 < len(parent_path):
        #     # if goal/next parent_node is not on same hierarchy as start
        #     if parent_path[i+1].unique_name != start_hierarchy[index-1]:
        #         goal_name = parent_path[i+1].unique_name + "_h_bridge"

        # parent_path = parent_node._plan(start_name, goal_name)

        # if "h_bridge" in parent_node.unique_name:
        #     continue

        print("----------------------------")
        print("Node name:", self.unique_name)
        print("Child graph:", self.get_childs("name"))
        print("Start node:", start_name)
        print("Goal node:", goal_name)
        child_path = self._plan(start_name, goal_name)

        print("path:", [node.unique_name for node in child_path])

        prev_parent = None
        hierarchy_level += 1
        for i, node in enumerate(child_path):
            if i+1 < len(child_path):
                next_parent = child_path[i+1]
            else:
                next_parent = None

            node._plan_recursive(start_hierarchy, goal_hierarchy, hierarchy_mask,
                                 hierarchy_level, prev_parent, next_parent)
            prev_parent = node

    def draw_child_graph(self, view_axis: int = 2):
        pos_index = [0, 1, 2]
        pos_index.remove(view_axis)
        pos_dict = {node: (node.pos_abs[pos_index[0]], node.pos_abs[pos_index[1]]) for node in self.get_childs()}

        nx.draw(self.child_graph,
                pos=pos_dict,
                labels=nx.get_node_attributes(self.child_graph, 'name'),
                with_labels=True)

        plt.show()


class SHGraph(SHNode):
    def __init__(self,  root_name: str, root_pos: Tuple[float, float, float]):
        self.root_graph: nx.Graph = nx.Graph()
        self.leaf_graph: nx.Graph = nx.Graph()
        self.root_graph.add_node(self, name=root_name)
        super().__init__(unique_name=root_name,
                         parent_node=self.root_graph[self], pos=root_pos, is_root=True)

    def add_child(self, hierarchy: List[str], name: str, pos: Tuple[float, float, float], is_leaf: bool = False, **data):
        self.get_child(hierarchy)._add_child(name, pos, is_leaf, **data,)
        if is_leaf:
            hierarchy.append(name)
            self.leaf_graph.add_node(self.get_child(hierarchy),
                                     name=name, **data)

    def add_connection(self, hierarchy_1: List[str], hierarchy_2: List[str], distance: Optional[float] = None, **data):
        add_hierarchy_bridge: bool = False
        for index, (name_1, name_2) in enumerate(zip(hierarchy_1, hierarchy_2)):
            if add_hierarchy_bridge:
                print("New hierarchy connection between:")
                print(name_1, hierarchy_2[index-1] + "_h_bridge", "in graph:", hierarchy_1[:index])
                print(name_2, hierarchy_1[index-1] + "_h_bridge", "in graph:", hierarchy_2[:index])
                is_leaf = False
                if name_1 == hierarchy_1[-1]:
                    node_1 = self.get_child(hierarchy_1)
                    is_leaf = node_1.is_leaf
                self.get_child(hierarchy_1[:index])._add_child(hierarchy_2[index-1] + "_h_bridge",
                                                               pos=(0, 0, 0), is_leaf=is_leaf, type="hierarchy_bridge")
                self.get_child(hierarchy_1[:index])._add_connection(
                    name_1, hierarchy_2[index-1] + "_h_bridge", distance, **data)
                self.get_child(hierarchy_2[:index])._add_child(hierarchy_1[index-1] + "_h_bridge",
                                                               pos=(0, 0, 0), is_leaf=is_leaf, type="hierarchy_bridge")
                self.get_child(hierarchy_2[:index])._add_connection(
                    name_2, hierarchy_1[index-1] + "_h_bridge", distance, **data)
            elif name_1 != name_2:
                print("New connection between:", name_1, name_2, "in graph:", hierarchy_1[:index])
                self.get_child(hierarchy_1[:index])._add_connection(name_1, name_2, distance, **data)
                add_hierarchy_bridge = True
            if name_1 == hierarchy_1[-1]:
                node_1 = self.get_child(hierarchy_1)
                if node_1.is_leaf:
                    node_2 = self.get_child(hierarchy_2)
                    if distance is None:
                        distance = self.get_euclidean_distance(node_1.pos, node_2.pos)
                    self.leaf_graph.add_edge(node_1, node_2, distance=distance, **data)

    def get_child(self, hierarchy: List[str]) -> SHNode:
        child = self
        for name in hierarchy:
            child = child._get_child(name)
        return child

    def draw_leaf_graph(self):
        node_xyz = np.array([node.pos_abs for node in self.leaf_graph.nodes()])
        edge_xyz = np.array([(u.pos_abs, v.pos_abs) for u, v in self.leaf_graph.edges])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*node_xyz.T, s=100, ec="w")

        for node in self.leaf_graph.nodes():
            ax.text(node.pos_abs[0], node.pos_abs[1], node.pos_abs[2],  node.unique_name, size=7, color='k')

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

    def create_graph_from_dict(self):
        pass

    def plan(self, start_hierarchy: List[str], goal_hierarchy: List[str]):
        # complete_path: List[SHNode] = []
        parent_path: List[SHNode] = []
        for index, (start_name, goal_name) in enumerate(zip(start_hierarchy, goal_hierarchy)):
            if index == 0:
                parent_path = self._plan(start_name, goal_name)
                continue
            for i, parent_node in enumerate(parent_path):
                # node : SHNode = node
                new_parent_path: List[SHNode] = []
                plan_start = start_name
                plan_goal = goal_name

                # if parent_node is not on same hierarchy as start
                if parent_node.unique_name != start_hierarchy[index-1]:
                    plan_start = parent_path[i-1].unique_name + "_h_bridge"

                # if not the last node in parent_path
                if i+1 < len(parent_path):
                    # if goal/next parent_node is not on same hierarchy as start
                    if parent_path[i+1].unique_name != start_hierarchy[index-1]:
                        plan_goal = parent_path[i+1].unique_name + "_h_bridge"

                new_parent_path = parent_node._plan(plan_start, plan_goal)

                if "h_bridge" in parent_node.unique_name:
                    continue

                print("----------------------------")
                print("parent name:", parent_node.unique_name)
                print("child graph:", parent_node.get_childs("name"))
                print("Start node:", plan_start)
                print("Goal node:", plan_goal)
                print("path:", [node.unique_name for node in new_parent_path])
            # parent_path = new_parent_path
        print("finished")

    def plan_recursive(self, start_hierarchy: List[str], goal_hierarchy: List[str]):

        hierarchy_mask = self.compare_hierarchy(start_hierarchy, goal_hierarchy)
        self._plan_recursive(start_hierarchy, goal_hierarchy, hierarchy_mask,
                             hierarchy_level=0, prev_parent=None, next_parent=None)

        # parent_path: List[SHNode] = self._plan(start_hierarchy[0], goal_hierarchy[0])

    def compare_hierarchy(self, hierarchy_1: List[str], hierarchy_2: List[str]) -> List[bool]:
        hierarchy_mask = [True]
        different_branch = False
        for start, goal in zip(hierarchy_1, hierarchy_2):
            if start != goal or different_branch:
                hierarchy_mask.append(False)
                different_branch = True
            else:
                hierarchy_mask.append(True)

        return hierarchy_mask


def main():
    # G = TestGraph(3)
    # G.create_example_data()
    # G.draw_single_level(level=0)
    # G.draw_multiple_level()

    # G = nx.Graph(name="LTC Campus", level=0, pos=(0, 0, 0))
    # child = nx.Graph()
    # G.add_node(child, name="Building F", level=1, parent=G, pos=(1, 1, 0))
    # G.add_node(nx.Graph(), name="Building A", level=1, parent=G, pos=(1, 0, 0))
    # print(G.nodes[child])
    # print([data["name"] for n, data in G.nodes(data=True)])

    G = SHGraph(root_name="LTC Campus", root_pos=(0, 0, 0))

    # G._add_child(name="Building F", pos=(-5, 0, 0))
    # G._get_child("Building F")._add_child(name="Floor 0", pos=(0, 0, 0))
    # G._get_child("Building F")._get_child("Floor 0")._add_child(name="Staircase", pos=(0, -1, 0))
    # G.add_connection("Building F", "Building A", name="path_1", type="path")

    G.add_child(hierarchy=[], name="Building A", pos=(0, 0, 0))
    G.add_child(hierarchy=[], name="Building B", pos=(5, 0, 0))
    G.add_child(hierarchy=[], name="Building C", pos=(5, 5, 0))
    G.add_child(hierarchy=[], name="Building D", pos=(0, 5, 0))
    G.add_child(hierarchy=[], name="Building E", pos=(-5, 5, 0))
    G.add_child(hierarchy=[], name="Building F", pos=(-5, 0, 0))
    G.add_child(hierarchy=["Building F"], name="Floor 0", pos=(0, 0, 0))
    G.add_child(hierarchy=["Building F"], name="Floor 1", pos=(0, 0, 1))
    G.add_child(hierarchy=["Building F"], name="Floor 2", pos=(0, 0, 2))
    G.add_child(hierarchy=["Building F"], name="Floor 3", pos=(0, 0, 3))
    G.add_child(hierarchy=["Building A"], name="Floor 0", pos=(0, 0, 0))
    G.add_child(hierarchy=["Building A"], name="Floor 1", pos=(0, 0, 1))
    G.add_child(hierarchy=["Building A"], name="Floor 2", pos=(0, 0, 2))
    G.add_child(hierarchy=["Building A"], name="Floor 3", pos=(0, 0, 3))
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="IRAS", pos=(0, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="xLab", pos=(1, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Kitchen", pos=(2, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Corridor", pos=(1, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 1"], name="Meeting Room", pos=(1, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="Lab", pos=(0, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="Workshop", pos=(1, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 0"], name="RoboEduLab", pos=(0, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Corridor", pos=(1, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Seminar Room 1", pos=(0, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Seminar Room 2", pos=(0, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Seminar Room 3", pos=(2, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Office", pos=(2, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 2"], name="Kitchen", pos=(3, 1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 3"], name="Staircase", pos=(0, -1, 0), is_leaf=True)
    G.add_child(hierarchy=["Building F", "Floor 3"], name="Office", pos=(1, 0, 0), is_leaf=True)
    G.add_child(hierarchy=["Building A", "Floor 1"], name="Cantina", pos=(1, 0, 0), is_leaf=True)

    # print(G.get_dict())
    # print(G.get_childs("name"))
    # [print(node.pos, node.pos_abs) for node in G._get_child("Building F").get_childs()]

    G.add_connection(["Building F", "Floor 0", "Staircase"],
                     ["Building F", "Floor 0", "Lab"], name="floor_door", type="door")
    G.add_connection(["Building F", "Floor 0", "Lab"],
                     ["Building F", "Floor 0", "Workshop"], name="door", type="door")
    G.add_connection(["Building F", "Floor 0", "Lab"],
                     ["Building F", "Floor 0", "RoboEduLab"], name="door", type="door")
    G.add_connection(["Building F", "Floor 0", "Staircase"],
                     ["Building F", "Floor 1", "Staircase"], distance=4.0, name="stair_F", type="stair")
    G.add_connection(["Building F", "Floor 1", "Staircase"],
                     ["Building F", "Floor 1", "Corridor"], name="floor_door", type="door")
    G.add_connection(["Building F", "Floor 1", "Corridor"],
                     ["Building F", "Floor 1", "IRAS"], name="open", type="open")
    G.add_connection(["Building F", "Floor 1", "Corridor"],
                     ["Building F", "Floor 1", "xLab"], name="open", type="open")
    G.add_connection(["Building F", "Floor 1", "Corridor"],
                     ["Building F", "Floor 1", "Meeting Room"], name="door", type="door")
    G.add_connection(["Building F", "Floor 1", "Corridor"],
                     ["Building F", "Floor 1", "Kitchen"], name="open", type="open")
    G.add_connection(["Building F", "Floor 1", "Staircase"],
                     ["Building F", "Floor 2", "Staircase"], distance=4.0, name="stair_F", type="stair")
    G.add_connection(["Building F", "Floor 2", "Staircase"],
                     ["Building F", "Floor 2", "Corridor"], name="floor_door", type="door")
    G.add_connection(["Building F", "Floor 2", "Corridor"],
                     ["Building F", "Floor 2", "Seminar Room 1"], name="door", type="door")
    G.add_connection(["Building F", "Floor 2", "Corridor"],
                     ["Building F", "Floor 2", "Seminar Room 2"], name="door", type="door")
    G.add_connection(["Building F", "Floor 2", "Corridor"],
                     ["Building F", "Floor 2", "Seminar Room 3"], name="door", type="door")
    G.add_connection(["Building F", "Floor 2", "Corridor"],
                     ["Building F", "Floor 2", "Office"], name="door", type="door")
    G.add_connection(["Building F", "Floor 2", "Corridor"],
                     ["Building F", "Floor 2", "Kitchen"], name="open", type="open")
    G.add_connection(["Building F", "Floor 2", "Staircase"],
                     ["Building F", "Floor 3", "Staircase"], distance=4.0, name="stair_F", type="stair")
    G.add_connection(["Building F", "Floor 3", "Staircase"],
                     ["Building F", "Floor 3", "Office"], name="floor_door", type="door")
    G.add_connection(["Building F", "Floor 1", "Kitchen"],
                     ["Building A", "Floor 1", "Cantina"], distance=10.0, name="terrace_door", type="door")

    # G.draw_child_graph()
    # G.get_child(["Building F"]).draw_child_graph(view_axis=0)
    # G.get_child(["Building F", "Floor 0"]).draw_child_graph()
    # G.get_child(["Building F", "Floor 1"]).draw_child_graph()
    # G.get_child(["Building F", "Floor 2"]).draw_child_graph()
    # G.get_child(["Building F", "Floor 3"]).draw_child_graph()

    # G.draw_leaf_graph()

    G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 1", "Cantina"])


if __name__ == "__main__":
    main()
