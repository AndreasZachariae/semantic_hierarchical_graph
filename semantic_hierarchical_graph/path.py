import itertools
from typing import Dict, List, Tuple, Union
import json
import numpy as np
import semantic_hierarchical_graph.utils as util


class SHMultiPaths():
    possibilities = []

    def __init__(self, parent):
        self.paths: List[Union['SHPath', 'SHMultiPaths']] = []
        self.parent = parent
        self.debug = True

    @property
    def distance(self):
        # return max(self.paths, key=lambda x: x.distance).distance
        return [p.distance for p in self.paths]

    @property
    def num_paths(self):
        return len(self.paths)

    def reduce_if_one(self):
        if self.num_paths == 1:
            return self.paths[0]
        return self

    def reduce_to_different_goals(self):

        if any([isinstance(path, SHMultiPaths) for path in self.paths]):
            return self

        start_pairs = [(p.start, p.goal) for p in self.paths]  # type: ignore
        unique_starts = set(start_pairs)

        different_paths = []
        for pair in unique_starts:
            same = [path for path, starts in zip(self.paths, start_pairs) if starts == pair]
            different_paths.append(min(same, key=lambda x: x.distance))

        # if this is omitted, all different possible paths on same graph are considered
        self.paths = different_paths

        if self.num_paths == 1:
            return self.paths[0]

        return self

    def add(self, path: Union['SHPath', 'SHMultiPaths']):
        self.paths.append(path)

    def to_dict(self, names: bool = False):
        return [path.to_dict(names) for path in self.paths]

    def to_json(self, file_path: str):
        dict = self.to_dict(names=True)
        with open(file_path, 'w') as outfile:
            json.dump(dict, outfile, indent=4, sort_keys=False)

    def _get_possibilities(self):
        SHMultiPaths.possibilities.append((self.parent.hierarchy, self.num_paths))
        [p._get_possibilities() for p in self.paths]

    def _get_single_comb(self, combination):
        num = combination[str(self.parent.hierarchy)]
        return self.paths[num]._get_single_comb(combination)

    @staticmethod
    def get_shortest_path(multi_path: Union['SHPath', 'SHMultiPaths']) -> Tuple[Dict, float]:

        multi_path._get_possibilities()
        # print(SHMultiPaths.possibilities)
        combinations = []
        for possibility in SHMultiPaths.possibilities:
            combinations.append([(str(possibility[0]), i) for i in range(possibility[1])])

        path_list = []
        for comb in itertools.product(*combinations):
            path_list.append(multi_path._get_single_comb(dict(comb)))

        shortest_path = min(path_list, key=lambda x: x[1])

        valid_paths = [path for path in path_list if path[1] != np.inf]
        print(f"Found paths: {len(path_list)}, valid: {len(valid_paths)}, shortest: {shortest_path[1]}")

        return shortest_path[0], shortest_path[1]


class SHPath():
    def __init__(self, start, goal, parent, distance):
        self.start = start
        self.goal = goal
        self.parent = parent
        self.parent_name = parent.unique_name
        self.path: List[Union['SHPath', 'SHMultiPaths']] = []
        self.distance = distance

    def add(self, path: Union['SHPath', 'SHMultiPaths']):
        self.path.append(path)

    def to_dict(self, names: bool = False):
        dict = {}
        for node in self.path:
            distance = str([round(d) for d in node.distance]) if isinstance(
                node.distance, list) else str(round(node.distance))
            key = node.parent.unique_name + " - " + distance if names else node.parent
            dict[key] = node.to_dict(names)

        return dict

    def to_json(self, file_path: str):
        dict = self.to_dict(names=True)
        with open(file_path, 'w') as outfile:
            json.dump(dict, outfile, indent=4, sort_keys=False)

    def _get_possibilities(self):
        [p._get_possibilities() for p in self.path]

    def _get_single_comb(self, combination):
        path = {}
        distance = self.distance
        prev_bridge = []

        for node in self.path:
            sub_path, sub_distance = node._get_single_comb(combination)

            if self.path.index(node) == 0:
                if node.parent.is_bridge:
                    prev_bridge = node.parent.bridge_to

            if len(sub_path) > 0:
                if list(sub_path)[0].bridge_to != prev_bridge:
                    # Additional check if prev parent was a bridge but the bridge_to list is missing the lowest level node
                    if not set(prev_bridge).issubset(set(list(sub_path)[0].bridge_to)):
                        # print("No valid path for", list(sub_path)[0].bridge_to, "!=", prev_bridge)
                        return {}, np.inf

                # print(list(sub_path)[0].bridge_to, "==", prev_bridge)

                if list(sub_path)[-1].is_bridge:
                    prev_bridge = list(sub_path)[-2].hierarchy
                else:
                    prev_bridge = list(sub_path)[-1].hierarchy

            path[node.parent] = sub_path
            distance += sub_distance

        return path, distance

    @staticmethod
    def save_path(path: Dict, file_path: str):
        try:
            path_names = util._map_names_to_nodes(path)
        except:
            path_names = path
        with open(file_path, 'w') as outfile:
            json.dump(path_names, outfile, indent=4, sort_keys=False)
