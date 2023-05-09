from typing import Dict, List, Optional, Tuple

from path_planner_suite.IPRRT import RRT
from semantic_hierarchical_graph.planners.planner_interface import PlannerInterface


class RRTPlanner(PlannerInterface):
    def __init__(self, room, config: Optional[Dict] = None):
        if config is None:
            config = dict()
            config["numberOfGeneratedNodes"] = 300
            config["testGoalAfterNumberOfNodes"] = 10
            config["smoothing_algorithm"] = "bechtold_glavina"
            config["smoothing_max_iterations"] = 100
            config["smoothing_max_k"] = 50

        super().__init__(room, config)

        self.name = "RRT"
        self.planner = RRT(self.collision_checker)

    @classmethod
    def around_point(cls, point: Tuple[float, float], max_dist: float, scene: List, config: Dict):
        obj = cls.__new__(cls)
        obj.name = "RRT"
        obj.config = config
        obj.collision_checker = cls._convert_from_point_to_IPCollisionChecker(obj, point, max_dist, scene)
        obj.planner = RRT(obj.collision_checker)
        return obj
