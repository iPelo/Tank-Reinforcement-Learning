from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple

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
    if d == Direction.E:
        return (1, 0)
    if d == Direction.S:
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
class Tank:
    x: int
    y: int
    dir: Direction
    cooldown: int = 0
    alive: bool = True
    team_id: str = "neutral"

@dataclass
class StepInfo:
    player_win: bool
    enemy_win: bool
    draw: bool
    player_hit: bool
    enemy_hit: bool
    steps: int
    phase: int
    agent_wins: Dict[str, bool] = field(default_factory=dict)
    agent_hits: Dict[str, bool] = field(default_factory=dict)
    agent_alive: Dict[str, bool] = field(default_factory=dict)
    team_wins: Dict[str, bool] = field(default_factory=dict)
    team_alive_counts: Dict[str, int] = field(default_factory=dict)
    winning_team: str | None = None
