# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""


class PlanerBase(object):

    def __init__(self, collisionChecker):
        """Base constructor

        Args:

            :environment: Reference to Environment

        """
        self._collisionChecker = collisionChecker

    def _checkStartGoal(self, startList, goalList):
        """Basic check for start and goal

        Args:

            :startList: list of start configurations
            :goalList: list of goal configurations

        """
        newStartList = list()
        for start in startList:
            if (len(start) != self._collisionChecker.getDim()):
                continue
            if self._collisionChecker.pointInCollision(start):
                continue
            newStartList.append(start)

        newGoalList = list()
        for goal in goalList:
            if (len(goal) != self._collisionChecker.getDim()):
                continue
            if self._collisionChecker.pointInCollision(goal):
                continue
            newGoalList.append(goal)

        if len(newStartList) == 0:
            raise Exception("No valid start")
        if len(newGoalList) == 0:
            raise Exception("No valid goal")

        return newStartList, newGoalList
