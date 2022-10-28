import networkx as nx
from typing import Dict, List, Tuple, Optional

from semantic_hierarchical_graph.node import SHNode
import semantic_hierarchical_graph.utils as util


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


class SHGraph(SHNode):
    def __init__(self,  root_name: str, root_pos: Tuple[float, float, float]):
        self.root_graph: nx.Graph = nx.Graph()
        self.leaf_graph: nx.Graph = nx.Graph()
        self.root_graph.add_node(self, name=root_name)
        super().__init__(unique_name=root_name,
                         parent_node=self.root_graph[self], pos=root_pos, is_root=True)

    def add_child(self, hierarchy: List[str], name: str, pos: Tuple[float, float, float], is_leaf: bool = False, **data):
        self.get_child(hierarchy)._add_child(name, pos, is_leaf, **data,)

        # Add leaf node to leaf_graph for visualization
        if is_leaf:
            hierarchy.append(name)
            self.leaf_graph.add_node(self.get_child(hierarchy),
                                     name=name, **data)

    def add_child_recursive(self):
        pass

    def add_connection_recursive(self, hierarchy_1: List[str], hierarchy_2: List[str], distance: Optional[float] = None, **data):
        if len(hierarchy_1) != len(hierarchy_2):
            raise ValueError("Hierarchies must have same length")

        hierarchy_mask = self._compare_hierarchy(hierarchy_1, hierarchy_2)
        self._add_connection_recursive(hierarchy_1[0], hierarchy_2[0], hierarchy_1, hierarchy_2, hierarchy_mask,
                                       hierarchy_level=0, distance=distance, **data)

        # Add leaf node connections to leaf_graph for visualization
        node_1 = self.get_child(hierarchy_1)
        if node_1.is_leaf:
            node_2 = self.get_child(hierarchy_2)
            if distance is None:
                distance = util.get_euclidean_distance(node_1.pos, node_2.pos)
            self.leaf_graph.add_edge(node_1, node_2, distance=distance, **data)

    def get_child(self, hierarchy: List[str]) -> SHNode:
        child = self
        for name in hierarchy:
            child = child._get_child(name)
        return child

    def create_graph_from_dict(self):
        pass

    # @util.timing
    def plan_recursive(self, start_hierarchy: List[str], goal_hierarchy: List[str]) -> Dict:
        if len(start_hierarchy) != len(goal_hierarchy):
            raise ValueError("Hierarchies must have same length")

        child_path: List[SHNode] = self._plan(start_hierarchy[0], goal_hierarchy[0])
        path_dict = {}
        path_dict[self] = self._plan_recursive(start_hierarchy[0], goal_hierarchy[0], start_hierarchy, goal_hierarchy, child_path,
                                               hierarchy_level=0)

        return path_dict

    # @util.timing
    def plan(self, start_hierarchy: List[str], goal_hierarchy: List[str]) -> List[SHNode]:
        if len(start_hierarchy) != len(goal_hierarchy):
            raise ValueError("Hierarchies must have same length")

        start_node = self.get_child(start_hierarchy)
        goal_node = self.get_child(goal_hierarchy)

        path_list = nx.shortest_path(self.leaf_graph,
                                     source=start_node,
                                     target=goal_node,
                                     weight="distance",
                                     method="dijkstra")

        return path_list  # type: ignore
