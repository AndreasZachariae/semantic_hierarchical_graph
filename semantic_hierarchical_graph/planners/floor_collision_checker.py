from shapely.geometry import Point, Polygon, LineString
import cv2
import numpy as np


class FloorCollisionChecker(object):

    def __init__(self, floor):
        self.scene = floor.ws_erosion
        shape = floor.ws_erosion.shape
        self.limits = [[0.0, shape[1]-1], [0.0, shape[0]-1]]

    def getDim(self):
        """ Return dimension of Environment (Shapely should currently always be 2)"""
        return 2

    def getEnvironmentLimits(self):
        """ Return limits of Environment"""
        return list(self.limits)

    def pointInCollision(self, pos):
        """ Return whether a configuration is
        inCollision -> True
        Free -> False """
        assert (len(pos) == self.getDim())
        if int(self.scene[round(pos[1]), round(pos[0])]) in [0, 1]:
            return True
        return False

    def lineInCollision(self, startPos, endPos):
        """ Check whether a line from startPos to endPos is colliding"""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())

        ws_erosion = self.scene.copy()

        pt1 = np.round(startPos).astype("int32")
        pt2 = np.round(endPos).astype("int32")
        cv2.line(ws_erosion, pt1, pt2, (-2), 1, cv2.LINE_4)

        collision_points = self.scene[np.where(ws_erosion == -2)]
        if any([pt in [0, 1] for pt in collision_points]):
            return True
        return False

    def drawObstacles(self, ax):
        raise NotImplementedError
