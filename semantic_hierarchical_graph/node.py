import networkx as nx
from typing import Dict, List, Tuple, TypeVar, Generic, Optional
# from typing_extensions import Self
import numpy as np

import semantic_hierarchical_graph.utils as util


T = TypeVar('T', bound='SHNode')
Position = Tuple[float, float, float]


class SHNode(Generic[T]):
    def __init__(self, unique_name: str, parent_node, pos: Position, is_root: bool = False, is_leaf: bool = False,
                 is_bridge: bool = False, bridge_to: List = [], type: str = "node"):
        self.unique_name: str = unique_name
        self.type: str = type
        self.is_root: bool = is_root
        self.is_leaf: bool = is_leaf
        self.is_bridge: bool = is_bridge
        self.bridge_to: List = bridge_to
        self.pos: Position = pos
        self.pos_abs: Position = tuple(np.add(pos, parent_node.pos_abs)) if not is_root else pos
        self.parent_node: SHNode = parent_node
        self.hierarchy: List[str] = parent_node.hierarchy + [unique_name] if not is_root else []
        self.child_graph: nx.Graph = nx.Graph()

    def add_child_by_name(self, name: str, pos: Position, is_leaf: bool = False,
                          is_bridge: bool = False, bridge_to: List = [], type: str = "node") -> T:
        # Create and add child of type SHNode with unique name on that level
        node: T = SHNode(unique_name=name, parent_node=self, pos=pos, is_leaf=is_leaf,
                         is_bridge=is_bridge, bridge_to=bridge_to, type=type)  # type: ignore
        self.add_child_by_node(node)
        return node

    def add_child_by_node(self, node: T):
        # Add child of type SHNode with unique name on that level
        self.child_graph.add_node(node, name=node.unique_name)

        # Add leaf node to leaf_graph for visualization
        if node.is_leaf and not node.is_bridge:
            leaf_graph = self._get_root_node().leaf_graph
            leaf_graph.add_node(node, name=node.unique_name)

    def _get_root_node(self):
        return self.parent_node._get_root_node()

    def add_connection_by_nodes(self, child_1: T, child_2: T, distance: Optional[float] = None, **data):
        color = "gray"
        if distance is None:
            distance = util.get_euclidean_distance(child_1.pos_abs, child_2.pos_abs)
            color = "black"
        self.child_graph.add_edge(child_1, child_2, distance=distance, color=color, **data)

        # Add leaf node connections to leaf_graph for visualization
        leaf_graph = self._get_root_node().leaf_graph
        if leaf_graph.has_node(child_1) and leaf_graph.has_node(child_2):
            leaf_graph.add_edge(child_1, child_2, distance=distance, color=color, **data)

    def add_connection_recursive_by_nodes(self, child_1: T, child_2: T, distance: Optional[float] = None, **data):
        root_node = self._get_root_node()
        root_node.add_connection_recursive(child_1.hierarchy, child_2.hierarchy, distance, **data)

    def _add_connection_by_names(self, child_name_1: str, child_name_2: str, distance: Optional[float] = None, **data):
        self.add_connection_by_nodes(self._get_child(child_name_1), self._get_child(child_name_2), distance, **data)

    def _add_connection_recursive(self, child_1_name, child_2_name, hierarchy_1: List[str], hierarchy_2: List[str], hierarchy_mask: List[bool],
                                  hierarchy_level: int, distance: Optional[float] = None, debug=False, **data):
        child_1 = self._get_child(hierarchy_1[hierarchy_level])

        # if childs are the same, they don't need a connection
        if hierarchy_mask[hierarchy_level] == True:
            child_1._add_connection_recursive(hierarchy_1[hierarchy_level+1], hierarchy_2[hierarchy_level+1], hierarchy_1, hierarchy_2,
                                              hierarchy_mask, hierarchy_level + 1, distance, **data)
            return

        # if node does not exist in graph, create new bridge_node
        if self._get_child(child_2_name, supress_error=True) is None:
            if debug:
                print("Add new bridge node:", child_2_name, "in graph:", hierarchy_1[:hierarchy_level])
            self.add_child_by_name(child_2_name, pos=(child_1.pos[0], child_1.pos[1], child_1.pos[2]+1),
                                   is_leaf=child_1.is_leaf, is_bridge=True, bridge_to=hierarchy_2, type="hierarchy_bridge")

        if debug:
            print("----------------------------")
            print("New connection between:", child_1_name, child_2_name, "in graph:", hierarchy_1[:hierarchy_level])
            print("Graph nodes:", self.get_childs("name"))

        # add connection between childs
        self._add_connection_by_names(child_1_name, child_2_name, distance, **data)

        # if child_1 is a leaf, no need to go deeper
        if child_1.is_leaf:
            if debug:
                print(child_1.unique_name, " is leaf")
            return

        # if childs are on the same parent graph but different, add bridges on each branch
        if hierarchy_level == 0 or hierarchy_mask[hierarchy_level-1] == True:
            child_1_bridge = self._get_hierarchy_bridge_name(hierarchy_1, hierarchy_mask, hierarchy_level)
            child_2 = self._get_child(hierarchy_2[hierarchy_level])
            child_2._add_connection_recursive(hierarchy_2[hierarchy_level+1], child_1_bridge, hierarchy_2, hierarchy_1,
                                              hierarchy_mask, hierarchy_level + 1, distance, **data)

        child_2_bridge = self._get_hierarchy_bridge_name(hierarchy_2, hierarchy_mask, hierarchy_level)
        child_1._add_connection_recursive(hierarchy_1[hierarchy_level+1], child_2_bridge, hierarchy_1, hierarchy_2,
                                          hierarchy_mask, hierarchy_level + 1, distance, **data)

    def _get_hierarchy_bridge_name(self, hierarchy: List[str], hierarchy_mask: List[bool], hierarchy_level: int):
        """ Return concated name of all graph levels where the branch is different 
            including the node name on the same level. """
        inverted_mask = [not elem for elem in hierarchy_mask][:hierarchy_level+2]
        relevant_hierarchy = hierarchy[:hierarchy_level+2]
        names = np.array(relevant_hierarchy)[inverted_mask]
        return str.join("_", names) + "_bridge"

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
        # Could be way more efficient if using hierarchy as hashable node instead of SHNode object
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

    def _plan(self, start_name: str, goal_name: str) -> List[T]:
        return nx.shortest_path(self.child_graph,
                                source=self._get_child(start_name),
                                target=self._get_child(goal_name),
                                weight="distance",
                                method="dijkstra")  # type: ignore

    def _plan_recursive(self, start_name: str, goal_name: str, start_hierarchy: List[str], goal_hierarchy: List[str],
                        child_path: List[T], hierarchy_level: int, debug=False) -> Dict:
        path_dict = {}
        for i, node in enumerate(child_path):

            # if node is leaf, no deeper planning
            if node.is_leaf:
                if debug:
                    print("Leaf reached:", node.unique_name)
                path_dict[node] = {}
                continue

            # if node is bridge, go to next in path
            if node.is_bridge:
                if debug:
                    print("Bridge reached:", node.unique_name)
                path_dict[node] = {}
                continue

            # if current node is not the start node, start from bridge
            if node.unique_name != start_name:
                child_start_name = self._get_bridge_node_name(node, child_path[i-1], hierarchy_level)
            else:
                child_start_name = start_hierarchy[hierarchy_level+1]

            # if current node is not the goal node, go to next bridge
            if node.unique_name != goal_name:
                child_goal_name = self._get_bridge_node_name(node, child_path[i+1], hierarchy_level)
            else:
                child_goal_name = goal_hierarchy[hierarchy_level+1]

            path = node._plan(child_start_name, child_goal_name)

            if debug:
                print("----------------------------")
                print("Node name:", node.unique_name, "in graph:", self.unique_name)
                print("Child graph:", node.get_childs("name"))
                print("Start node:", child_start_name)
                print("Goal node:", child_goal_name)
                print("path:", [node.unique_name for node in path])

            path_dict[node] = node._plan_recursive(path[0].unique_name, path[-1].unique_name, start_hierarchy, goal_hierarchy,
                                                   path, hierarchy_level + 1)

        return path_dict

    def _get_bridge_node_name(self, node: T, bridge_node: T, hierarchy_level: int) -> str:
        bridge_found = []
        for n in node.get_childs():
            if n.is_bridge:
                if bridge_node.is_bridge and (n.bridge_to == bridge_node.bridge_to):
                    bridge_found.append(n.unique_name)
                    continue
                elif n.bridge_to[hierarchy_level] == bridge_node.unique_name:
                    bridge_found.append(n.unique_name)

        if len(bridge_found) == 1:
            return bridge_found[0]
        elif len(bridge_found) > 1:
            raise ValueError("Multiple bridges found")
        raise ValueError("No next child as bridge to other branch can be found")
