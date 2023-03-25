from typing import Dict, Optional

from path_planner_suite.IPBasicPRM import BasicPRM
from semantic_hierarchical_graph.planners.planner_interface import PlannerInterface


class PRMPlanner(PlannerInterface):
    def __init__(self, room, config: Optional[Dict] = None):
        if config is None:
            config = dict()
            config["radius"] = 50
            config["numNodes"] = 300
            config["smoothing_max_iterations"] = 100
            config["smoothing_max_k"] = 50

        super().__init__(room, config)

        self.name = "PRM"
        self.planner = BasicPRM(self.collision_checker)
