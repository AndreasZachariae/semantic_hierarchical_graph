# coding: utf-8

"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""
from shapely.geometry import Point, LineString
from shapely.plotting import plot_polygon

from path_planner_suite.IPPerfMonitor import IPPerfMonitor


class CollisionChecker(object):

    def __init__(self, scene, limits=[[0.0, 22.0], [0.0, 22.0]], statistic=None):
        self.scene = scene
        self.limits = limits

    def getDim(self):
        """ Return dimension of Environment (Shapely should currently always be 2)"""
        return 2

    def getEnvironmentLimits(self):
        """ Return limits of Environment"""
        return list(self.limits)

    @IPPerfMonitor
    def pointInCollision(self, pos):
        """ Return whether a configuration is
        inCollision -> True
        Free -> False """
        assert (len(pos) == self.getDim())
        point = Point(pos[0], pos[1])
        for key, value in self.scene.items():
            if value.intersects(point):
                # Change for compatibility with SHGraph
                # points directly on the edge are not considered as collision
                if not value.touches(point):
                    return True
        return False

    @IPPerfMonitor
    def lineInCollision(self, startPos, endPos):
        """ Check whether a line from startPos to endPos is colliding"""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())

        line = LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])
        for key, value in self.scene.items():
            if value.intersects(line):
                # Change for compatibility with SHGraph
                # points directly on the edge are not considered as collision
                if not value.touches(line):
                    return True
        return False

    def drawObstacles(self, ax):
        for key, value in self.scene.items():
            plot_polygon(value, ax=ax, add_points=False, color="red", alpha=0.8)
