from functools import wraps
import json
import collections.abc
from time import time
import timeit
from typing import List, Tuple


def get_euclidean_distance(pos_1: Tuple, pos_2: Tuple) -> float:
    return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2 + (pos_1[2] - pos_2[2]) ** 2) ** 0.5


def path_dict_to_leaf_path_list(path: dict):
    leaf_path = []
    for node, dict in path.items():
        if node.is_leaf and not "h_bridge" in node.unique_name:
            leaf_path.append(node)
        else:
            leaf_path.extend(path_dict_to_leaf_path_list(dict))
    return leaf_path


def path_dict_to_child_path_list(path: dict, child_hierarchy: List[str]):
    child_path = []
    if len(path) == 1:
        path = list(path.values())[0]

        # if path from root_node is asked
        if len(child_hierarchy) == 0:
            [child_path.append(k) for k, v in path.items()]
            return child_path

    for node, dict in path.items():
        if node.unique_name == child_hierarchy[0]:
            if len(child_hierarchy) > 1:
                child_path.extend(path_dict_to_child_path_list(dict, child_hierarchy[1:]))
            else:
                [child_path.append(k) for k, v in dict.items()]

    return child_path


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


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.6f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap


def planning_time():
    # python -m timeit -r 10 -s 'from semantic_hierarchical_graph.main import main; G = main()'
    #                           'G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])'
    setup_code = '''from semantic_hierarchical_graph.main import main
                    G = main()
                 '''

    test_code = '''G.plan_recursive(["Building F", "Floor 0", "Lab"], ["Building A", "Floor 0", "Entrance"])'''
    result = timeit.Timer(test_code, setup_code, globals=globals()).repeat(10)

    print(result)
