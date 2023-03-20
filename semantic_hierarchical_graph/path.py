from typing import List
import json
import numpy as np


class SHMultiPaths():
    def __init__(self):
        self.paths: List['SHPath'] = []
        self.debug = True

    @property
    def num_paths(self):
        return len(self.paths)

    @property
    def shortest_path(self):
        min_path = min(self.paths, key=lambda x: x.distance)
        if self.debug:
            if len(self.paths) > 1:
                print("----------------------------")
                for path in self.paths:
                    mark = ">" if path is min_path else ""
                    print(mark + path.start, "-", path.goal, "=", path.distance)

        return min_path

    def add(self, path: 'SHPath'):
        self.paths.append(path)


class SHPath():
    def __init__(self, start, goal, parent, distance):
        self.start = start
        self.goal = goal
        self.parent = parent
        self.path: List['SHPath'] = []
        self.distance = distance

    def add(self, path: 'SHPath'):
        self.path.append(path)
        self.distance += path.distance

    def _to_dict(self, names: bool):
        return {node.parent.unique_name + " - " + str(round(node.distance)) if names else node.parent: node._to_dict(names) for node in self.path}

    def to_dict(self, names: bool = False):
        return {self.parent.unique_name if names else self.parent: self._to_dict(names)}

    def to_json(self, file_path: str):
        dict = self.to_dict(names=True)
        with open(file_path, 'w') as outfile:
            json.dump(dict, outfile, indent=4, sort_keys=False)

    # def get_shortest_path(self, path=None):
    #     if path is None:
    #         path = self.all_paths
    #     if isinstance(path, list):
    #         if len(path) == 1:
    #             return_path, distance = self.get_shortest_path(path[0]["path"])
    #             return return_path, path["distance"] + distance
    #         else:
    #             shortest_path = min(path, key=lambda x: x["distance"])
    #             for p in path:
    #                 return self.get_shortest_path(p["path"]), p["distance"]
    #             # return self.get_shortest_path(shortest_path["path"])

    #     elif isinstance(path, dict):
    #         # if path == {}:
    #         #     self.reached_leaf = True
    #         #     return {}
    #         if list(path.keys()) == ["path", "distance"]:
    #             if path["path"] == {}:
    #                 return {}, path["distance"]
    #             path, distance = self.get_shortest_path(path["path"])
    #             return path, path["distance"] + distance
    #         return_path = {}
    #         distance = 0
    #         for k, v in path.items():
    #             p, d = self.get_shortest_path(v)
    #             return_path[k] = p
    #             distance += d

    #         return return_path, distance
    #     else:
    #         raise SHGException()
