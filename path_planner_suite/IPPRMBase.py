# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPerfMonitor import IPPerfMonitor

try:
    from IPPlanerBase import PlanerBase
except:
    from templates.IPPlanerBase import PlanerBase

import random


class PRMBase(PlanerBase):

    def __init__(self, collChecker):
        super(PRMBase, self).__init__(collChecker)

    def _getRandomPosition(self):
        limits = self._collisionChecker.getEnvironmentLimits()
        pos = [random.uniform(limit[0], limit[1]) for limit in limits]
        return pos

    @IPPerfMonitor
    def _getRandomFreePosition(self):
        pos = self._getRandomPosition()
        while self._collisionChecker.pointInCollision(pos):
            pos = self._getRandomPosition()
        return pos
