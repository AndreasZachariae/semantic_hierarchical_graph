import itertools
from typing import List, Union
import matplotlib.pyplot as plt
from shapely import MultiLineString
from shapely.plotting import plot_polygon, plot_line
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import nearest_points, transform
from semantic_hierarchical_graph.types.exceptions import SHGGeometryError
from semantic_hierarchical_graph.types.vector import Vector


class Environment():
    def __init__(self, room_id: int):
        self.room_id: int = room_id
        self.scene: List[Polygon] = []
        self.path: List[LineString] = []
        self.room_bbox: List[int] = [0, 0, 0, 0]
        self.floor_bbox: List[int] = [0, 0, 0, 0]

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

    def find_shortest_connection(self, pos):
        """ Find the shortest path from pos to any path that is not in collision """
        if not self.path:
            return None, None
        if not isinstance(pos, Point):
            pos = Point(pos[0], pos[1])
        closest_path = min(self.path, key=lambda x: x.distance(pos))
        closest_point: Point = nearest_points(closest_path, pos)[0]
        connection = self.get_valid_connection(pos, closest_point)
        if connection is not None:
            return connection, closest_path
        return None, closest_path

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
            if isinstance(line, MultiLineString):
                remove_lines.append(line)
                for line2 in line.geoms:
                    for i in range(len(line2.coords) - 1):
                        new_lines.append(LineString([line2.coords[i], line2.coords[i + 1]]))
            else:
                if len(line.coords) > 2:
                    remove_lines.append(line)
                    for i in range(len(line.coords) - 1):
                        new_lines.append(LineString([line.coords[i], line.coords[i + 1]]))
        self.path = list((set(self.path) - set(remove_lines)) | set(new_lines))

    def split_path_at_intersections(self):
        """ Split all paths at intersections.
        Assuming that all paths are two-point lines."""
        already_cut = dict()
        for line1, line2 in itertools.combinations(self.path, 2):
            # Check if lines cross each other in X shape
            if line1.intersects(line2):
                point = line1.intersection(line2)
                if not point.geom_type == "Point":
                    if len(point.coords) > 2 or len(line1.coords) > 2 or len(line2.coords) > 2:
                        raise SHGGeometryError(
                            "Intersection is not a POINT or a LINE with max two points but a", point.geom_type)
                    if line1.distance(Point(line2.coords[0])) < 0.01:
                        p2 = Point(line2.coords[0])
                    else:
                        p2 = Point(line2.coords[-1])
                    if line2.distance(Point(line1.coords[0])) < 0.01:
                        p1 = Point(line1.coords[0])
                    else:
                        p1 = Point(line1.coords[-1])

                    self._split_path(line1, p2, already_cut)
                    self._split_path(line2, p1, already_cut)
                    continue
                # print("Intersection:", point)

                self._split_path(line1, point, already_cut)
                self._split_path(line2, point, already_cut)

            # Check if lines touch each other in T shape
            else:
                for point1 in line1.coords:
                    if point1 not in line2.coords and line2.distance(Point(point1)) < 0.01:
                        self._split_path(line2, Point(point1), already_cut)

                for point2 in line2.coords:
                    if point2 not in line1.coords and line1.distance(Point(point2)) < 0.01:
                        self._split_path(line1, Point(point2), already_cut)

        add_lines = {cut for cuts in already_cut.values() for cut in cuts}

        self.path = list((set(self.path) - set(already_cut.keys())) | add_lines)

    def _split_path(self, line, point, already_cut):
        if point.coords[0] in (line.coords[0], line.coords[-1]):
            return
        if line in already_cut:
            # print("Already cut")
            for cut_line in already_cut[line]:
                # Account for dec precision error
                if cut_line.distance(point) < 0.01:
                    cut_results = self.cut(cut_line, point)
                    already_cut[line].remove(cut_line)
                    already_cut[line].extend(cut_results)
                    break
        else:
            results = self.cut(line, point)
            already_cut[line] = results

    def cut(self, line, point):
        # Cuts a line in two at a point
        distance = line.project(point)
        if distance <= 0.0 or distance >= line.length:
            return [LineString(line)]
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [
                    LineString(coords[:i+1]),
                    LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])]

    def clear_bridge_edges(self, bridge_edges: List):
        for i, wall in enumerate(self.scene):
            for edge in bridge_edges:
                if len(edge) == 1:
                    print("single edge point in room ", self.room_id, edge)
                    wall = wall.difference(Point(edge[0]).buffer(4))
                else:
                    wall = wall.difference(LineString(edge).buffer(4, cap_style="flat"))

            if isinstance(wall, GeometryCollection):
                wall = [geom for geom in wall.geoms if isinstance(geom, Polygon)]
                self.scene[i] = wall[0]
                for j in range(1, len(wall)):
                    self.scene.append(wall[j])
            else:
                self.scene[i] = wall

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_yaxis()  # type: ignore
        ax.set_aspect("equal")
        for value in self.scene:
            plot_polygon(value, ax=ax, add_points=False, color="red", alpha=0.8)

        for value in self.path:
            plot_line(value, ax=ax, add_points=False, color="blue", alpha=0.5)

        plt.show()
