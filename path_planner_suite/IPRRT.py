from scipy.spatial import cKDTree
from IPPRMBase import PRMBase
from IPPerfMonitor import IPPerfMonitor
import numpy as np

import networkx as nx
import random


class RRT(PRMBase):

    def __init__(self, _collChecker):
        """
        _collChecker: the collision checker interface
        """
        super(RRT, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.lastGeneratedNodeNumber = 0

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """

        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:
            config["numberOfGeneratedNodes"] = 500 
            config["testGoalAfterNumberOfNodes"]  = 10
        """
        # 0. reset
        self.graph.clear()
        self.lastGeneratedNodeNumber = 0

        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

        # 2. add start and goal to graph
        self.graph.add_node(self.lastGeneratedNodeNumber, pos=checkedStartList[0])
        self.lastGeneratedNodeNumber += 1

        while self.lastGeneratedNodeNumber < config["numberOfGeneratedNodes"]:

            posList = list(nx.get_node_attributes(self.graph, 'pos').values())
            kdTree = cKDTree(posList)

            if (self.lastGeneratedNodeNumber % config["testGoalAfterNumberOfNodes"]) == 0:
                # print "testing goal"
                result = kdTree.query(checkedGoalList[0], k=1)
                if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'], checkedGoalList[0]):
                    self.graph.add_node("goal", pos=checkedGoalList[0])
                    self.graph.add_edge(result[1], "goal")
                    mapping = {0: 'start'}
                    self.graph = nx.relabel_nodes(self.graph, mapping)

                    # return nx.shortest_path(self.graph,"start","goal")
                    try:
                        path = nx.shortest_path(self.graph, "start", "goal")
                    except:
                        return []
                    return path

            pos = self._getRandomFreePosition()

            # for every node in graph find nearest neigbhours
            result = kdTree.query(pos, k=1)
            if None:
                raise Exception("Something went wrong regarding nearest neighbours")

            start = np.array(self.graph.nodes[result[1]]['pos'])
            end = np.array(pos)
            newPos = 0.5 * (end-start) + start
            if not self._collisionChecker.lineInCollision(self.graph.nodes[result[1]]['pos'], newPos):
                self.graph.add_node(self.lastGeneratedNodeNumber, pos=newPos)
                self.graph.add_edge(result[1], self.lastGeneratedNodeNumber)
                self.lastGeneratedNodeNumber += 1

        return []
