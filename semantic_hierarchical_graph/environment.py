import itertools
from typing import List, Union
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import nearest_points, transform, split
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError
from semantic_hierarchical_graph.types.vector import Vector


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
        if self._reverse_geom(path) in self.path:
            return
        self.path.append(path)

    def _in_collision_with_shape(self, shape1, shape2) -> bool:
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

    def _in_collision(self, shape) -> bool:
        """
        Return whether a shape is in collision with any shape in the scene.
        """
        for value in self.scene:
            if self._in_collision_with_shape(value, shape):
                return True
        return False

    def _is_orthogonal_to_grid(self, line: LineString) -> bool:
        """ Check whether a line is orthogonal to the grid """
        x_grid = Vector(1, 0)
        y_grid = Vector(0, 1)
        line_vector = Vector(line.coords[1][0] - line.coords[0][0], line.coords[1][1] - line.coords[0][1])
        return x_grid.dot(line_vector) == 0 or y_grid.dot(line_vector) == 0

    def get_valid_connection(self, point1: Point, point2: Point, mode: str = "", polygon=None) -> Union[LineString, None]:
        if point1 == point2:
            return None
        connection = LineString([point1, point2])
        if self._in_collision(connection):
            # print("Connection in collision between", point1, point2)
            return None
        else:
            if mode == "orthogonal" and not self._is_orthogonal_to_grid(connection):
                return None
            if mode == "without_polygon":
                if isinstance(polygon, MultiPolygon) or isinstance(polygon, Polygon):
                    if self._in_collision_with_shape(polygon, connection):
                        return None
            return connection

    def find_shortest_connection(self, pos, max_attempts=1):
        """ Find the shortest path from pos to any path that is not in collision """
        if not isinstance(pos, Point):
            point = Point(pos[0], pos[1])
        tmp_path = self.path.copy()
        for attempts in range(max_attempts):
            closest_path = min(tmp_path, key=lambda x: x.distance(point))
            closest_point: Point = nearest_points(closest_path, point)[0]
            connection = self.get_valid_connection(point, closest_point)
            if connection is not None:
                return connection, closest_path
            tmp_path.remove(closest_path)
        return None, None

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

    def _reverse_geom(self, geom):
        def _reverse(x, y, z=None):
            if z:
                return x[::-1], y[::-1], z[::-1]
            return x[::-1], y[::-1]

        return transform(_reverse, geom)

    def remove_duplicate_paths(self):
        """ Remove all duplicate or covered paths"""
        self.path = list(set(self.path))
        remove_lines = []
        for line in self.path:
            if line in remove_lines:
                # print("not in path", line)
                continue
            # if line.length == 0:
            #     # print("line length is 0", line)
            #     remove_lines.append(line2)
            #     continue
            for line2 in self.path:
                if line is line2:
                    # print("is same", line2)
                    continue
                # if line == self._reverse_geom(line2):
                #     # print("is reverse", line2)
                #     remove_lines.append(line2)
                #     continue
                if line.covers(line2):
                    # print("is covered", line2)
                    remove_lines.append(line2)
        self.path = list(set(self.path) - set(remove_lines))

    def split_multipoint_lines(self):
        """ Split multipoint lines in two point segments"""
        new_lines = []
        remove_lines = []
        for line in self.path:
            if len(line.coords) > 2:
                remove_lines.append(line)
                for i in range(len(line.coords) - 1):
                    new_lines.append(LineString([line.coords[i], line.coords[i + 1]]))
        self.path = list((set(self.path) - set(remove_lines)) | set(new_lines))

    def split_path_at_intersections(self):
        """ Split all paths at intersections"""
        already_cut = dict()
        for line1, line2 in itertools.combinations(self.path, 2):
            if line1.intersects(line2):
                point = line1.intersection(line2)
                if not point.geom_type == "Point":
                    raise SHGGeometryError("Intersection is not a POINT but a", point.geom_type)
                # print("Intersection:", point)

                self._split_path(line1, point, already_cut)
                self._split_path(line2, point, already_cut)

        add_lines = {cut for cuts in already_cut.values() for cut in cuts}

        self.path = list((set(self.path) - set(already_cut.keys())) | add_lines)

    def _split_path(self, line, point, already_cut):
        if point.coords[0] in (line.coords[0], line.coords[-1]):
            return
        if line in already_cut:
            # print("Already cut")
            for cut in already_cut[line]:
                # Account for dec precision error
                if cut.distance(point) < 1e-8:
                    cut_results = split(cut, point)
                    already_cut[line].remove(cut)
                    already_cut[line].extend(cut_results.geoms)
                    break
        else:
            results = split(line, point)
            already_cut[line] = list(results.geoms)

    def clear_bridge_edges(self, bridge_edges: List):
        walls = self.scene[0]
        for edge in bridge_edges:
            if len(edge) == 1:
                print("single edge point in room ", self.room_id, edge)
                walls = walls.difference(Point(edge[0]).buffer(4))
            else:
                walls = walls.difference(LineString(edge).buffer(4, cap_style="flat"))
        self.scene[0] = walls

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_yaxis()  # type: ignore
        ax.set_aspect("equal")
        for value in self.scene:
            plot_polygon(value, ax=ax, add_points=False, color="red", alpha=0.8)

        for value in self.path:
            plot_line(value, ax=ax, add_points=False, color="blue", alpha=0.5)

        plt.show()
