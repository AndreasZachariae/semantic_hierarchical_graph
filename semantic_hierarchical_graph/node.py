import networkx as nx
from typing import Dict, List, Tuple, TypeVar, Generic, Optional
import numpy as np

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

    def _add_connection_recursive(self, child_1_name, child_2_name, hierarchy_1: List[str], hierarchy_2: List[str], hierarchy_mask: List[bool],
                                  hierarchy_level: int, distance: Optional[float] = None, **data):
        print("----------------------------")
        print("Childs:", child_1_name, child_2_name)
        child_1 = self._get_child(hierarchy_1[hierarchy_level])

        # if childs are the same, they don't need a connection
        if hierarchy_mask[hierarchy_level] == True:
            child_1._add_connection_recursive(hierarchy_1[hierarchy_level+1], hierarchy_2[hierarchy_level+1], hierarchy_1, hierarchy_2,
                                              hierarchy_mask, hierarchy_level + 1, distance, **data)
            return

        # if node does not exist in graph, create new bridge_node
        if self._get_child(child_2_name, supress_error=True) is None:
            print("Add new bridge node:", child_2_name, "in graph:", hierarchy_1[:hierarchy_level])
            self._add_child(child_2_name, pos=(0, 0, 0), is_leaf=child_1.is_leaf, type="hierarchy_bridge")

        print("New connection between:", child_1_name, child_2_name, "in graph:", hierarchy_1[:hierarchy_level])
        print("Graph nodes:", self.get_childs("name"))
        self._add_connection(child_1_name, child_2_name, distance, **data)

        # if child_1 is a leaf, no need to go deeper
        if child_1.is_leaf:
            print(child_1.unique_name, " is leaf")
            return

        # if childs are on the same graph but different, add bridges on each branch
        if hierarchy_level == 0 or hierarchy_mask[hierarchy_level-1] == True:
            child_1_bridge = self._get_hierarchy_bridge_name(hierarchy_1, hierarchy_mask, hierarchy_level)
            child_2 = self._get_child(hierarchy_2[hierarchy_level])
            child_2._add_connection_recursive(hierarchy_2[hierarchy_level+1], child_1_bridge, hierarchy_2, hierarchy_1,
                                              hierarchy_mask, hierarchy_level + 1, distance, **data)

        child_2_bridge = self._get_hierarchy_bridge_name(hierarchy_2, hierarchy_mask, hierarchy_level)
        child_1._add_connection_recursive(hierarchy_1[hierarchy_level+1], child_2_bridge, hierarchy_1, hierarchy_2,
                                          hierarchy_mask, hierarchy_level + 1, distance, **data)

    def _get_hierarchy_bridge_name(self, hierarchy: List[str], hierarchy_mask: List[bool], hierarchy_level: int):
        bridge_name = ""
        for i, node_name in enumerate(hierarchy[:hierarchy_level]):
            if hierarchy_mask[i] == False:
                bridge_name += node_name + "_"
        if hierarchy_level < len(hierarchy):
            bridge_name += hierarchy[hierarchy_level] + "_"
        bridge_name += "h_bridge"
        return bridge_name

    def _compare_hierarchy(self, hierarchy_1: List[str], hierarchy_2: List[str]) -> List[bool]:
        hierarchy_mask = []
        different_branch = False
        for start, goal in zip(hierarchy_1, hierarchy_2):
            if start != goal or different_branch:
                hierarchy_mask.append(False)
                different_branch = True
            else:
                hierarchy_mask.append(True)

        return hierarchy_mask

    def _get_child(self, name: str, supress_error: bool = False) -> T:
        # Could be way more efficient if using hierarchy_list as hashable node instead of SHNode object
        # Drawback: node.pos etc is not that easy, has to get the data dict all the time with graph.nodes(data=True)
        for node in self.child_graph.nodes:
            if node.unique_name == name:
                return node
        if not supress_error:
            raise ValueError("Child with name {} not found".format(name))
        else:
            return None  # type: ignore

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

    def _plan_recursive(self, start_name: str, goal_name: str, start_hierarchy: List[str], goal_hierarchy: List[str], child_path: List[T],
                        hierarchy_level: int, bridge_start=None, bridge_goal=None) -> Dict:

        # start_name = start_hierarchy[hierarchy_level]
        # goal_name = goal_hierarchy[hierarchy_level]

        # child_path = self._plan(start_name, goal_name)

        path_dict = {}
        same_hierarchy_paths: Dict[T, List[T]] = {}
        for i, node in enumerate(child_path):

            # if node is leaf, no deeper planning
            if node.is_leaf:
                print("Leaf reached:", node.unique_name)
                path_dict[node] = {}
                continue

            # if node is bridge, go to next in path
            if "_h_bridge" in node.unique_name:
                print("Bridge reached:", node.unique_name)
                path_dict[node] = {}
                continue

            # if current node is not the start node, start from bridge
            if node.unique_name != start_name:
                if "_h_bridge" in child_path[i-1].unique_name:
                    child_start_name = child_path[i-1].unique_name[:-9] + "_" + str(bridge_start) + "_h_bridge"
                else:
                    child_start_name = child_path[i-1].unique_name + "_h_bridge"
            else:
                child_start_name = start_hierarchy[hierarchy_level+1]

            # if current node is not the goal node, go to next bridge
            if node.unique_name != goal_name:
                if "_h_bridge" in child_path[i+1].unique_name:
                    child_goal_name = child_path[i+1].unique_name[:-9] + "_" + str(bridge_goal) + "_h_bridge"
                else:
                    child_goal_name = child_path[i+1].unique_name + "_h_bridge"
            else:
                child_goal_name = goal_hierarchy[hierarchy_level+1]

            path = node._plan(child_start_name, child_goal_name)
            same_hierarchy_paths[node] = path

            print("----------------------------")
            print("Node name:", node.unique_name, "in graph:", self.unique_name)
            print("Child graph:", node.get_childs("name"))
            print("Start node:", child_start_name)
            print("Goal node:", child_goal_name)
            print("path:", [node.unique_name for node in path])

            # node._plan_recursive(start_name, goal_name, start_hierarchy, goal_hierarchy, new_hierarchy, hierarchy_mask,
            #                      hierarchy_level, prev_parent, next_parent)

        print("same_hierarchy_paths:", {key.unique_name: [
              item.unique_name for item in value] for key, value in same_hierarchy_paths.items()})

        for node, path in same_hierarchy_paths.items():
            start_name = path[0].unique_name
            goal_name = path[-1].unique_name
            if not path[0].is_leaf:
                if "_h_bridge" in goal_name:
                    bridge_goal = same_hierarchy_paths[self._get_child(goal_name[:-9])][1].unique_name
                if "_h_bridge" in start_name:
                    bridge_start = same_hierarchy_paths[self._get_child(start_name[:-9])][-2].unique_name

            child_path_dict = node._plan_recursive(start_name, goal_name, start_hierarchy, goal_hierarchy, path,
                                                   hierarchy_level + 1, bridge_start, bridge_goal)
            path_dict[node] = child_path_dict

        return path_dict
