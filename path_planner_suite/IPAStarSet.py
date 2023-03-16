# coding: utf-8
"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

It is based on the slides given during the course, so please **read the information in theses slides first**

Remark: The code is a slightly modified version of AStar, whithout reopening of nodes, when once in the closed list.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import copy
import networkx as nx
import heapq
import numpy as np
from scipy.spatial.distance import euclidean, cityblock

from path_planner_suite.IPPlanerBase import PlanerBase
from path_planner_suite.IPPerfMonitor import IPPerfMonitor


class AStar(PlanerBase):
    def __init__(self, collChecker):
        """Contructor:

        Initialize all necessary members"""

        super(AStar, self).__init__(collChecker)
        self.graph = nx.DiGraph()  # = CloseList
        self.openList = []  # (<value>, <node>)

        self.goal = set()
        self.goalFound = False

        self.limits = self._collisionChecker.getEnvironmentLimits()
        self.dim = self._collisionChecker.getDim()

        self.w = 0.5
        return

    def _getNodeID(self, pos):
        """Compute a hashable unique identifier based on the position"""

        nodeId = "-"
        for i in pos:
            nodeId += str(i)+"-"
        return nodeId

    def _getPosFromNodeID(self, nodeId):
        """Compute the position from the hashable unique identifier"""

        pos = []
        for i in nodeId.split("-"):
            if i != "":
                pos.append(int(i))
        return pos

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:

            config["w"] = 0.5
            config["heuristic"] = "euclid"

        """
        # 0. reset
        self.graph.clear()
        self.openList = []
        self.goalFound = False

        # 1. check start and goal whether collision free (s. BaseClass)
        # checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        checkedStartSet, checkedGoalSet = self._checkStartGoalSet(startList, goalList)

        # 2.
        self.w = config["w"]
        self.heuristic = config["heuristic"]

        self.goal = checkedGoalSet
        for start in checkedStartSet:
            self._addGraphNode(start)

        currentBestName = self._getBestNodeName()
        breakNumber = 0
        while currentBestName:
            if breakNumber > config["max_iterations"]:
                print(f"Planning interrupted due to over {config['max_iterations']} iterations")
                break

            breakNumber += 1

            currentBest = self.graph.nodes[currentBestName]

            if currentBestName in self.goal:
                self.solutionPath = []
                self._collectPath(currentBestName, self.solutionPath)
                mapping = {self.solutionPath[0]: 'start',
                           self.solutionPath[-1]: 'goal'}
                self.graph = nx.relabel_nodes(self.graph, mapping)
                self.solutionPath[0] = "start"
                self.solutionPath[-1] = "goal"
                self.goalFound = True
                break

            currentBest["status"] = 'closed'
            if self._collisionChecker.pointInCollision(currentBest["pos"]):
                currentBest['collision'] = 1
                if len(self.openList) == 0:
                    break
                currentBestName = self._getBestNodeName()
                continue
            self.graph.nodes[currentBestName]['collision'] = 0

            # handleNode merges with former expandNode
            self._handleNode(currentBestName)

            # No new nodes could be added, so no solution exists
            if len(self.openList) == 0:
                break

            currentBestName = self._getBestNodeName()

        if self.goalFound:
            return self.solutionPath
        else:
            return []

    def _insertNodeNameInOpenList(self, nodeName):
        """Get an existing node stored in graph and put it in the OpenList"""
        heapq.heappush(self.openList, (self._evaluateNode(nodeName), nodeName))

    def _addGraphNode(self, nodeID, fatherName=None):
        """Add a node based on the position into the graph. Attention: Existing node is overwritten!"""
        self.graph.add_node(nodeID, pos=self._getPosFromNodeID(nodeID), status='open', g=0)

        if fatherName != None:
            self.graph.add_edge(nodeID, fatherName)
            self.graph.nodes[nodeID]["g"] = self.graph.nodes[fatherName]["g"] + 1

        self._insertNodeNameInOpenList(nodeID)

    def _setLimits(self, lowLimit, highLimit):
        """ Sets the limits of the investigated search space """
        assert (len(lowLimit) == len(highLimit) == self.dim)
        self.limits = list()
        for i in range(self.dim):
            self.limits.append([lowLimit[i], highLimit[i]])
        return

    def _getBestNodeName(self):
        """ Returns the best name of best node """
        return heapq.heappop(self.openList)[1]

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """Generates possible successor positions in all dimensions"""
        result = []
        node = self.graph.nodes[nodeName]
        for i in range(len(node["pos"])):
            for u in [-1, 1]:
                newPos = copy.copy(node["pos"])
                newPos[i] += u
                if not self._inLimits(newPos):
                    continue
                try:
                    # Do not do reopening! If node already in graph do not add it... Concequences?
                    self.graph.nodes[self._getNodeID(newPos)]
                    continue
                except:
                    pass

                self._addGraphNode(self._getNodeID(newPos), nodeName)

        return result

    @IPPerfMonitor
    def _computeHeuristicValue(self, nodeName):
        """ Computes Heuristic Value: Manhattan Distance """

        dist = np.inf
        node = self.graph.nodes[nodeName]
        for goal in self.goal:
            if self.heuristic == "euclidean":
                new_dist = euclidean(self._getPosFromNodeID(goal), node["pos"])
            else:
                new_dist = cityblock(self._getPosFromNodeID(goal), node["pos"])
            if new_dist < dist:
                dist = new_dist
        return dist

    @IPPerfMonitor
    def _evaluateNode(self, nodeName):
        node = self.graph.nodes[nodeName]
        return self.w * self._computeHeuristicValue(nodeName) + (1 - self.w) * node["g"]

    def _collectPath(self, nodeName, solutionPath):

        fathers = list(self.graph.successors(nodeName))
        # print len(fathers)
        if len(fathers) == 1:
            self._collectPath(fathers[0], solutionPath)
        elif len(fathers) == 0:
            solutionPath.append(nodeName)
            return
        else:
            raise Exception("not suitable numbers of fathers = {}.... please check".format(len(fathers)))
        solutionPath.append(nodeName)
        return

    def _inLimits(self, pos):
        result = True
        for i, limit in enumerate(self.limits):
            if pos[i] < limit[0] or pos[i] > limit[1]:
                result = False
                break
        return result

    def _checkStartGoalSet(self, startList, goalList):
        """Basic check for start and goal
        This is a copy from the method in the base class but adjusted for sets.

        Args:

            :startList: list of start configurations
            :goalList: list of goal configurations

        """
        newStartSet = set()
        for start in startList:
            if (len(start) != self._collisionChecker.getDim()):
                continue
            if self._collisionChecker.pointInCollision(start):
                continue
            newStartSet.add(self._getNodeID(start))

        newGoalSet = set()
        for goal in goalList:
            if (len(goal) != self._collisionChecker.getDim()):
                continue
            if self._collisionChecker.pointInCollision(goal):
                continue
            newGoalSet.add(self._getNodeID(goal))

        if len(newStartSet) == 0:
            raise Exception("No valid start")
        if len(newGoalSet) == 0:
            raise Exception("No valid goal")

        return newStartSet, newGoalSet
