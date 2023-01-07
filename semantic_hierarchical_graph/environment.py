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
        """
        Return whether a point is
        in collision -> True
        Free -> False
        Touching (same boundary points but no same interior points) is not considered as collision.
        """
        for value in self.scene:
            if value.intersects(Point(pos[0], pos[1])):
                if value.touches(Point(pos[0], pos[1])):
                    return False
                else:
                    return True
        return False

    def line_in_collision(self, start_pos, end_pos):
        """
        Check whether a line from start_pos to end_pos is colliding.
        Touching (same boundary points but no same interior points) is not considered as collision.
        """
        if isinstance(start_pos, Point):
            start_pos = start_pos.coords._coords[0]
        if isinstance(end_pos, Point):
            end_pos = end_pos.coords._coords[0]
        for value in self.scene:
            if value.intersects(LineString([(start_pos[0], start_pos[1]), (end_pos[0], end_pos[1])])):
                if value.touches(LineString([(start_pos[0], start_pos[1]), (end_pos[0], end_pos[1])])):
                    return False
                else:
                    return True
        return False

    def find_shortest_connection(self, pos):
        """ Find the shortest path from pos to any path that is not in collision """
        closest_path = min(self.path, key=lambda x: x.distance(Point(pos[0], pos[1])))
        closest_point: Point = nearest_points(closest_path, Point(pos[0], pos[1]))[0]
        if self.line_in_collision(closest_point.coords._coords[0], (pos[0], pos[1])):
            print("Connection in collision")
            print(closest_point)
            return None
        else:
            connection = LineString([closest_point, Point(pos[0], pos[1])])
            # self.add_path(connection)
            return connection

    def find_all_shortest_connections(self):
        new_connections = []
        for path in self.path:
            # print("Path: ", path)
            rest_path = list(self.path)
            # for exclude_path in exclude:
            rest_path.remove(path)
            if len(rest_path) == 0:
                return new_connections
            for other_path in rest_path:
                # print("Other path: ", other_path)
                closest_point_path = Point(min(path.coords, key=lambda x: other_path.distance(Point(x[0], x[1]))))
                # print("Closest point path: ", closest_point_path)
                closest_point_other_path: Point = nearest_points(other_path, closest_point_path)[0]
                # print("Closest point other path: ", closest_point_other_path)
                if self.line_in_collision(closest_point_path, closest_point_other_path):
                    print("Connection in collision")
                else:
                    connection = LineString([closest_point_path, closest_point_other_path])
                    # print("Connection: ", connection)
                    new_connections.append(connection)

        return new_connections

    def clear_bridge_nodes(self, bridge_points: List):
        walls = self.scene[0]
        for point in bridge_points:
            self.scene[0] = walls.difference(Point(point).buffer(2))

    def clear_bridge_edges(self, bridge_edges: List):
        walls = self.scene[0]
        for edge in bridge_edges:
            self.scene[0] = walls.difference(LineString(edge).buffer(2, cap_style="flat"))

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        for value in self.scene:
            plot_polygon(value, ax=ax, add_points=False, color="red", alpha=0.8)

        for value in self.path:
            plot_line(value, ax=ax, add_points=False, color="blue", alpha=0.8)

        plt.show()
