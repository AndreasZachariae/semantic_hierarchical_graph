from typing import List
import numpy as np
from semantic_hierarchical_graph.collision_checker import CollisionChecker
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_points


class Environment(CollisionChecker):
    def __init__(self, name: str, limits: np.ndarray, scene: List = []):
        self.name = name
        self.limits = limits
        super().__init__(scene)

    def add_obstacle(self, obstacle):
        self.scene.append(obstacle)

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        print(len(self.scene))
        for value in self.scene:
            print(value)
            # patch = PolygonPatch(value, facecolor="red", alpha=0.8, zorder=2)
            # ax.add_patch(patch)
            plot_polygon(value, ax=ax, facecolor="red", alpha=0.8, zorder=2)
        plt.show()
