from typing import Tuple
import numpy as np


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_tuple(cls, pos: Tuple):
        return cls(pos[0], pos[1])

    @classmethod
    def from_two_points(cls, pos1: Tuple, pos2: Tuple):
        return cls(pos2[0] - pos1[0], pos2[1] - pos1[1])

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def norm(self):
        return self.dot(self)**0.5

    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)

    def perp(self):
        return Vector(1, -self.x / self.y)

    def angle(self, other):
        if self.norm() == 0 or other.norm() == 0:
            return 0
        return np.arctan2(self.x * other.y - self.y * other.x, self.x * other.x + self.y * other.y)
    
    def angle_to_grid(self):
        return self.angle(Vector(1, 0))

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __str__(self):
        return f'({self.x}, {self.y})'


class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_tuple(cls, pos: Tuple):
        return cls(pos[0], pos[1], pos[2])

    @classmethod
    def from_two_points(cls, pos1: Tuple, pos2: Tuple):
        return cls(pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2])

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return self.dot(self)**0.5

    def normalized(self):
        norm = self.norm()
        return Vector3D(self.x / norm, self.y / norm, self.z / norm)

    def cross(self, other):
        return Vector3D(self.y * other.z - self.z * other.y,
                        self.z * other.x - self.x * other.z,
                        self.x * other.y - self.y * other.x)

    def angle(self, other):
        if self.norm() == 0 or other.norm() == 0:
            return 0
        return np.arccos(self.dot(other) / (self.norm() * other.norm()))

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'
