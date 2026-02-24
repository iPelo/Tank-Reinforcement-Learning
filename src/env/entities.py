from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

class Direction(IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3

def turn_left(d: Direction) -> Direction:
    return Direction((int(d) - 1) % 4)

def turn_right(d:Direction) -> Direction:
    return Direction((int(d) + 1) % 4)

def dir_to_vec(d: Direction) -> Tuple[int, int]:
    if d == Direction.N:
        return (0, -1)
    elif d == Direction.E:
        return (1, 0)
    elif d == Direction.S:
        return (0, 1)
    return (-1, 0)

class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FWD = 3
    BWD = 4
    SHOOT = 5

@dataclass
class Target:
    x: int
    y: int

@dataclass
class StepInfo:
    success: bool
    hit: bool
    steps: int
    phase: int