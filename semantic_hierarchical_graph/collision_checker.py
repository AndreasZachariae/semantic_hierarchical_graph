
from typing import List
from shapely.geometry import Point, Polygon, LineString


class CollisionChecker():

    def __init__(self, scene: List):
        self.scene = scene

    def point_in_collision(self, pos):
        """ Return whether a point is
        in collision -> True
        Free -> False """
        for value in self.scene:
            if value.intersects(Point(pos[0], pos[1])):
                return True
        return False

    def line_in_collision(self, start_pos, end_pos):
        """ Check whether a line from startPos to endPos is colliding"""
        for value in self.scene:
            if value.intersects(LineString([(start_pos[0], start_pos[1]), (end_pos[0], end_pos[1])])):
                return True
        return False
