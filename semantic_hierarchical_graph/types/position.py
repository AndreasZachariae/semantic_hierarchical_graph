from typing import Optional, Tuple, Union, List
from semantic_hierarchical_graph.types.exceptions import SHGIndexError


class Position():
    def __init__(self, x: float, y: float, z: float, rz: Optional[float] = None) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.rz = rz

    @classmethod
    def from_iter(cls, pos: Union[Tuple, List]) -> 'Position':
        if len(pos) == 2:
            return cls(pos[0], pos[1], 0.0, None)
        elif len(pos) == 3:
            return cls(pos[0], pos[1], pos[2], None)
        else:
            return cls(pos[0], pos[1], pos[2], pos[3])

    @classmethod
    def from_map_frame(cls, pose: Tuple, map_origin: Tuple, resolution: float, map_shape: Tuple[int, int]) -> 'Position':
        x = int((pose[0] - map_origin[0]) / resolution)
        y = map_shape[0] - int((pose[1] - map_origin[1]) / resolution)
        return cls(x, y, 0.0)

    @property
    def xy(self) -> Tuple:
        return (self.x, self.y)

    @property
    def xyz(self) -> Tuple:
        return (self.x, self.y, self.z)

    def to_name(self) -> str:
        return str((round(self.x), round(self.y)))

    def to_map_frame(self, map_origin: Tuple, resolution: float, map_shape: Tuple[int, int]) -> Tuple:
        x = self.x * resolution + map_origin[0]
        y = (map_shape[0] - self.y) * resolution + map_origin[1]
        return (x, y, self.rz)

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

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))


def convert_map_pos_to_hierarchy(map, map_x, map_y, watershed):
    x, y = Position.from_map_frame((map_x, map_y),
                                   (map.info.origin.position.x, map.info.origin.position.y),
                                   map.info.resolution,
                                   (map.info.height, map.info.width)).xy
    room_str = "room_"+str(watershed[y, x])
    return [room_str, (x, y)]
