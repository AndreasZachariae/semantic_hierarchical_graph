from functools import wraps
import json
import collections.abc
from time import time
import timeit
from typing import Dict, List, Union
import gc
import sys


def round_up(n) -> int:
    # multiplier = 10 ** decimals
    # return math.ceil(n * multiplier) / multiplier
    return int(-(-n // 1))


def path_to_list(path: Union[List, Dict], relevant_hierarchy: List[str], with_names: bool = False, is_leaf: bool = False) -> List:
    """ Return list of the given path on the given hierarchy level.
        relevant_hierarchy must be a list or empty [] for leaf graph
    """
    if isinstance(path, dict):
        if is_leaf:
            path_list = _path_dict_to_leaf_path_list(path)
        else:
            path_list = _path_dict_to_child_path_list(path, relevant_hierarchy)
    else:
        path_list: List = path
    if with_names:
        path_list = map_names_to_nodes(path_list)  # type: ignore
    return path_list


def _path_dict_to_leaf_path_list(path: Dict):
    leaf_path = []
    for node, dict in path.items():
        if node.is_leaf and not "bridge" in node.unique_name:
            leaf_path.append(node)
        else:
            leaf_path.extend(_path_dict_to_leaf_path_list(dict))
    return leaf_path


def _path_dict_to_child_path_list(path: Dict, child_hierarchy: List[str]):
    # if path from root_node is asked
    if len(child_hierarchy) == 0:
        return list(path.keys())

    for node, dict in path.items():
        if node.unique_name == child_hierarchy[0]:
            if len(child_hierarchy) > 1:
                return _path_dict_to_child_path_list(dict, child_hierarchy[1:])
            else:
                return list(dict.keys())
    return []


def map_names_to_nodes(obj):
    if isinstance(obj, collections.abc.Mapping):
        return {k.unique_name: map_names_to_nodes(v) for k, v in obj.items()}  # type: ignore
    elif isinstance(obj, list):
        return [map_names_to_nodes(elem) for elem in obj]
    else:
        return obj.unique_name


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


def get_obj_size(obj):

    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def find_memory_leaks():
    import tracemalloc
    tracemalloc.start()

    path = [999999] * 1000000
    del path
    gc.collect()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
