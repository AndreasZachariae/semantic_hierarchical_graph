from typing import Tuple, Any
import yaml
from semantic_hierarchical_graph.utils import round_up


class Parameter():
    def __init__(self, path: str, is_floor: bool = True):
        self.params: dict[str, Any] = self.load_params(path)
        if is_floor:
            self.params["safety_distance"] = self.get_safety_distance(
                self.params["base_size"], self.params["safety_margin"])

    def load_params(self, path: str):
        with open(path, "r") as file:
            params = {}
            try:
                params = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(e)
        return params

    def get_safety_distance(self, base_size: Tuple[int, int], safety_margin: int) -> int:
        """
        Parameters
        ----------
        base_size : Tuple[int, int]
            (width from side to side, length in drive direction)
        safety_margin : int
            distance around robot as: (safety_margin + base_size[0] + safety_margin)

        Returns
        -------
        int
            distance from imaginary point robot
        """
        # TODO: extend to circle and with trailer
        return round_up(base_size[0]/2 + safety_margin)
