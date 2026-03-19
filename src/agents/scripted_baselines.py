from __future__ import annotations

from src.env.entities import Action, Direction


def chase_target(dx: int, dy: int, current_dir: Direction) -> Action:
    if abs(dx) >= abs(dy):
        goal_dir = Direction.E if dx > 0 else Direction.W
    else:
        goal_dir = Direction.S if dy > 0 else Direction.N

    if current_dir == goal_dir:
        return Action.FWD

    left_steps = (int(current_dir) - int(goal_dir)) % 4
    right_steps = (int(goal_dir) - int(current_dir)) % 4
    return Action.LEFT if left_steps <= right_steps else Action.RIGHT
