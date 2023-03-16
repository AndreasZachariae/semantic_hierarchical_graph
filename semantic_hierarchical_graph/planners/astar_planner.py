from typing import Dict, List, Optional, Tuple

from path_planner_suite.IPAStarSet import AStar
from semantic_hierarchical_graph.planners.planner_interface import PlannerInterface


class AStarPlanner(PlannerInterface):
    def __init__(self, room, config: Optional[Dict] = None):
        if config is None:
            config = dict()
            config["heuristic"] = 'euclidean'
            config["w"] = 0.5
            config['max_iterations'] = 100000
            config["smoothing_iterations"] = 50
            config["smoothing_max_k"] = 20
            config["smoothing_epsilon"] = 0.5
            config["smoothing_variance_window"] = 10
            config["smoothing_min_variance"] = 0.0

        super().__init__(room, config)

        self.name = "AStar"
        self.planner = AStar(self.collision_checker)

    @classmethod
    def around_point(cls, point: Tuple[float, float], max_dist: float, scene: List, config: Dict):
        obj = cls.__new__(cls)
        obj.name = "AStar"
        obj.config = config
        obj.collision_checker = cls._convert_from_point_to_IPCollisionChecker(obj, point, max_dist, scene)
        obj.planner = AStar(obj.collision_checker)
        return obj
