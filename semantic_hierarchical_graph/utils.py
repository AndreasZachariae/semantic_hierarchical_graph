import json
import collections.abc
from typing import List, Tuple


def get_euclidean_distance(pos_1: Tuple, pos_2: Tuple) -> float:
    return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2 + (pos_1[2] - pos_2[2]) ** 2) ** 0.5


def path_to_leaf_path(path: dict):
    leaf_path = []
    for node, dict in path.items():
        if node.is_leaf and not "h_bridge" in node.unique_name:
            leaf_path.append(node)
        else:
            leaf_path.extend(path_to_leaf_path(dict))
    return leaf_path


def map_names_to_nodes(obj):
    if isinstance(obj, collections.abc.Mapping):
        return {k.unique_name: map_names_to_nodes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [map_names_to_nodes(elem) for elem in obj]
    else:
        return obj.unique_name


def save_dict_to_json(dict, file_path: str, convert_to_names: bool = True):
    if convert_to_names:
        dict = map_names_to_nodes(dict)
    with open(file_path, 'w') as outfile:
        json.dump(dict, outfile, indent=4, sort_keys=False)
