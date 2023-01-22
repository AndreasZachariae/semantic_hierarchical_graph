import itertools
from typing import Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import nearest_points, transform
from semantic_hierarchical_graph.vector import Vector


class Environment():
    def __init__(self, room_id: int):
        self.room_id: int = room_id
        self.scene: List[Polygon] = []
        self.path: List[LineString] = []

    def add_obstacle(self, obstacle):
        if obstacle not in self.scene:
            self.scene.append(obstacle)

    def add_path(self, path):
        if path.length == 0:
            return
        if path in self.path:
            return
        if self.reverse_geom(path) in self.path:
            return
        self.path.append(path)

    def in_collision_with_shape(self, shape1, shape2) -> bool:
        """
        Check whether a shape is colliding with another shape.
        Touching (same boundary points but no same interior points) is not considered as collision.
        """
        if shape1.intersects(shape2):
            if shape1.touches(shape2):
                return False
            else:
                return True
        return False

    def in_collision(self, shape) -> bool:
        """
        Return whether a shape is in collision with any shape in the scene.
        """
        for value in self.scene:
            if self.in_collision_with_shape(value, shape):
                return True
        return False

    def is_orthogonal_to_grid(self, line: LineString) -> bool:
        """ Check whether a line is orthogonal to the grid """
        x_grid = Vector(1, 0)
        y_grid = Vector(0, 1)
        line_vector = Vector(line.coords[1][0] - line.coords[0][0], line.coords[1][1] - line.coords[0][1])
        return x_grid.dot(line_vector) == 0 or y_grid.dot(line_vector) == 0

    def get_valid_connection(self, point1: Point, point2: Point, mode: str = "", polygon=None) -> Union[LineString, None]:
        if point1 == point2:
            return None
        connection = LineString([point1, point2])
        if self.in_collision(connection):
            # print("Connection in collision between", point1, point2)
            return None
        else:
            if mode == "orthogonal" and not self.is_orthogonal_to_grid(connection):
                return None
            if mode == "without_polygon":
                if isinstance(polygon, MultiPolygon) or isinstance(polygon, Polygon):
                    if self.in_collision_with_shape(polygon, connection):
                        return None
            return connection

    def find_shortest_connection(self, pos):
        """ Find the shortest path from pos to any path that is not in collision """
        if not isinstance(pos, Point):
            point = Point(pos[0], pos[1])
        closest_path = min(self.path, key=lambda x: x.distance(point))
        closest_point: Point = nearest_points(closest_path, point)[0]
        return self.get_valid_connection(closest_point, point)

    def find_all_shortest_connections(self, mode: str, polygon=None):
        """ Find all shortest connections between all shapes in path that are not in collision """
        new_connections = []
        for path, other_path in itertools.permutations(self.path, 2):
            closest_point_path = Point(min(path.coords, key=lambda x: other_path.distance(Point(x[0], x[1]))))
            closest_point_other_path: Point = nearest_points(other_path, closest_point_path)[0]
            connection = self.get_valid_connection(closest_point_path, closest_point_other_path, mode, polygon)
            if connection is not None:
                new_connections.append(connection)

        return new_connections

    def find_all_vertex_connections(self, mode: str, polygon=None):
        """ Find all connections between all vertices in path that are not in collision """
        new_connections = []
        for path, other_path in itertools.combinations(self.path, 2):
            for point in path.coords:
                # closest_point_other_path: Point = nearest_points(other_path, Point(point))[0]
                # connection = self.get_connection(Point(point), closest_point_other_path, mode, polygon)
                # if connection is not None:
                #     new_connections.append(connection)
                for other_point in other_path.coords:
                    connection = self.get_valid_connection(Point(point), Point(other_point), mode, polygon)
                    if connection is not None:
                        new_connections.append(connection)

        return new_connections

    def reverse_geom(self, geom):
        def _reverse(x, y, z=None):
            if z:
                return x[::-1], y[::-1], z[::-1]
            return x[::-1], y[::-1]

        return transform(_reverse, geom)

    def clean_path(self):
        """ Remove all duplicate or covered paths """
        # print(len(self.path))
        self.path = list(set(self.path))
        removed_lines = []
        for line in self.path:
            if line in removed_lines:
                # print("not in path", line)
                continue
            # if line.length == 0:
            #     # print("line length is 0", line)
            #     self.path.remove(line)
            #     continue
            for line2 in self.path:
                if line is line2:
                    # print("is same", line2)
                    continue
                # if line == self.reverse_geom(line2):
                #     # print("is reverse", line2)
                #     self.path.remove(line2)
                #     removed_lines.append(line2)
                #     continue
                if line.covers(line2):
                    # print("is covered", line2)
                    # self.path.remove(line2)
                    removed_lines.append(line2)
        self.path = list(set(self.path) - set(removed_lines))
        # print(len(self.path))

    def clear_bridge_nodes(self, bridge_points: List):
        walls = self.scene[0]
        for point in bridge_points:
            self.scene[0] = walls.difference(Point(point).buffer(2))

    def clear_bridge_edges(self, bridge_edges: List):
        walls = self.scene[0]
        for edge in bridge_edges:
            self.scene[0] = walls.difference(LineString(edge).buffer(3, cap_style="flat"))

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        for value in self.scene:
            plot_polygon(value, ax=ax, add_points=False, color="red", alpha=0.8)

        for value in self.path:
            plot_line(value, ax=ax, add_points=False, color="blue", alpha=0.5)

        plt.show()
