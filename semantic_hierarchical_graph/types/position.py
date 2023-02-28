from typing import Tuple, Union, List
from semantic_hierarchical_graph.types.exceptions import SHGIndexError


class Position():
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_iter(cls, pos: Union[Tuple, List]) -> 'Position':
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0.0)
        return cls(pos[0], pos[1], pos[2])

    @classmethod
    def convert_to_grid(cls, pos: Union[Tuple, List, 'Position'], grid_size: float) -> 'Position':
        # TODO: Convert meters from map frame to pixels in grid map
        resolution = grid_size  # [m/px]
        if not isinstance(pos, Position):
            pos = Position.from_iter(pos)
        return pos

    @property
    def xy(self) -> Tuple:
        return (self.x, self.y)

    @property
    def xyz(self) -> Tuple:
        return (self.x, self.y, self.z)

    def get_tuple(self) -> Tuple:
        return (self.x, self.y, self.z)

    def to_name(self) -> str:
        return str((round(self.x), round(self.y)))

    def distance(self, other_pos: Union[Tuple, List, 'Position']) -> float:
        if not isinstance(other_pos, Position):
            other_pos = Position.from_iter(other_pos)
        return ((self.x - other_pos.x) ** 2 + (self.y - other_pos.y) ** 2 + (self.z - other_pos.z) ** 2) ** 0.5

    def __add__(self, other_pos: Union[Tuple, List, 'Position']) -> 'Position':
        if not isinstance(other_pos, Position):
            other_pos = Position.from_iter(other_pos)
        return Position(self.x + other_pos.x, self.y + other_pos.y, self.z + other_pos.z)

    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        else:
            raise SHGIndexError('Position index out of range')
