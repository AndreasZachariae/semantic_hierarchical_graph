from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points


class Environment():
    def __init__(self, name: Any, limits: np.ndarray):
        self.name = name
        self.limits = limits
        self.scene: List[Polygon] = []
        self.path: List[LineString] = []

    def add_obstacle(self, obstacle):
        self.scene.append(obstacle)

    def add_path(self, path):
        self.path.append(path)

    def point_in_collision(self, pos):
        """ Return whether a point is
        in collision -> True
        Free -> False """
        for value in self.scene:
            if value.intersects(Point(pos[0], pos[1])):
                return True
        return False

    def line_in_collision(self, start_pos, end_pos):
        """ Check whether a line from start_pos to end_pos is colliding"""
        for value in self.scene:
            if value.intersects(LineString([(start_pos[0], start_pos[1]), (end_pos[0], end_pos[1])])):
                return True
        return False

    def add_shortest_connection(self, pos):
        """ Find the shortest path from pos to any path that is not in collision """
        closest_path = min(self.path, key=lambda x: x.distance(Point(pos[0], pos[1])))
        closest_point: Point = nearest_points(closest_path, Point(pos[0], pos[1]))[0]
        # if self.line_in_collision(closest_point.coords._coords[0], (pos[0], pos[1])):
        #     print("Connection in collision")
        connection = LineString([closest_point, Point(pos[0], pos[1])])
        self.add_path(connection)

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        for value in self.scene:
            plot_polygon(value, ax=ax, add_points=False, color="red", alpha=0.8)

        for value in self.path:
            plot_line(value, ax=ax, add_points=False, color="blue", alpha=0.8)

        plt.show()
