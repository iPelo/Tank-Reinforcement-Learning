from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import numpy as np
from .entities import Action, Direction, StepInfo, Tank, Target, dir_to_vec, turn_left, turn_right
from .map_gen import generate_walls

Coord = Tuple[int, int]

@dataclass
class EnvState:
    tank: Tank
    target: Target
    walls: Set[Coord]
    steps: int
    phase: int

class TankEnv:

    def __init__(
        self,
        w: int = 15,
        h: int = 15,
        max_steps: int = 200,
        seed: int = 0,
        wall_density: float = 0.12,
        cooldown_steps: int = 5,
        ray_limit:Optional[int] = None,
    ) -> None:
        self.w = int(w)
        self.h = int(h)
        self.max_steps = int(max_steps)
        self.wall_density = float(wall_density)
        self.cooldown_steps = int(cooldown_steps)
        self.ray_limit = int(ray_limit) if ray_limit is not None else max(self.w, self.h)

        self.rng = np.random.default_rng(seed)
        self.state: Optional[EnvState] = None

    def reset(self, phase: int = 0) -> np.ndarray:
        phase = int(phase)

        while True:
            tank = Tank(
                x=int(self.rng.integers(1, self.w - 1)),
                y=int(self.rng.integers(1, self.h - 1)),
                dir=Direction(int(self.rng.integers(0,4))),
                cooldown=0,
            )
            target = Target(
                x=int(self.rng.integers(1, self.w - 1)),
                y=int(self.rng.integers(1, self.h - 1)),
            )

            if (tank.x, tank.y) == (target.x, target.y):
                continue

            forbidden = {
                (target.x, target.y),
                (tank.x, tank.y),
                (tank.x + 1, tank.y),
                (tank.x - 1, tank.y),
                (tank.x, tank.y + 1),
                (tank.x, tank.y - 1),
            }

            walls = generate_walls(self.w, self.h, self.wall_density, self.rng, forbidden=forbidden)

            if((tank.x, tank.y) in walls or (target.x, target.y) in walls):
                continue

            self.state = EnvState(tank=tank, target=target, walls=walls, steps=0 ,phase=phase)
            break

        return self._get_obs()

    def step(self,action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        s = self.state
        s.steps += 1
        act = Action(int(action))
        hit = False

        if s.tank.cooldown > 0:
            s.tank.cooldown -= 1

        if act == Action.LEFT:
            s.tank.dir = turn_left(s.tank.dir)
        elif act == Action.RIGHT:
            s.tank.dir = turn_right(s.tank.dir)
        elif act == Action.FWD:
            self._move_tank(True)
        elif act == Action.BWD:
            self._move_tank(False)

        success = False
        done = False

        if s.phase == 0:
            if (s.tank.x, s.tank.y) == (s.target.x, s.target.y):
                success = True
                done = True

        if s.steps >= self.max_steps:
            done = True

        reward = 1.0 if success else -0.01

        info = StepInfo(success=success, hit=hit, steps=s.steps, phase=s.phase)
        return self._get_obs(), float(reward), bool(done), {"info": info}

    def _in_bounds(self, x:int,y:int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def _is_wall(self, x: int, y: int) -> bool:
        if not self._in_bounds(x,y):
            return True
        assert self.state is not None
        return (x,y) in self.state.walls

    def _move_tank(self, forward: bool) -> None:
        assert self.state is not None
        dx, dy = dir_to_vec(self.state.tank.dir)
        if not forward:
            dx, dy = -dx, -dy

        nx = self.state.tank.x + dx
        ny = self.state.tank.y + dy

        if self._is_wall(nx, ny):
            return

        self.state.tank.x = nx
        self.state.tank.y = ny

    def _raycast_wall_dist(self, origin: Coord, direction: Coord, limit: int) -> int:
        assert self.state is not None
        ox, oy = origin
        dx, dy = direction
        dist = 0
        x, y = ox + dx, oy + dy

        while dist < limit:
            if self._is_wall(x, y):
                return dist
            dist += 1
            x += dx
            y += dy

        return limit

    def _raycast_target_dist(self, origin: Coord, direction: Coord, limit: int) -> int:
        assert self.state is not None
        ox, oy = origin
        dx, dy = direction
        dist = 0
        x, y = ox + dx, oy + dy

        while dist < limit:
            if self._is_wall(x,y):
                return -1
            if (x,y) == (self.state.target.x, self.state.target.y):
                return dist
            dist += 1
            x += dx
            y += dy

        return -1

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        t = self.state.tank

        fwd = dir_to_vec(t.dir)
        back = (-fwd[0], -fwd[1])
        left = dir_to_vec(turn_left(t.dir))
        right = dir_to_vec(turn_right(t.dir))
        dirs = [fwd, back, left, right]

        limit =  self.ray_limit
        origin = (t.x, t.y)

        wall_d = [self._raycast_wall_dist(origin, d, limit) for d in dirs]
        tgt_d = [self._raycast_target_dist(origin, d, limit) for d in dirs]

        wall_norm = [d / limit for d in wall_d]
        tgt_norm = [(-1.0 if d < 0 else d / limit) for d in tgt_d]

        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_onehot[int(t.dir)] = 1.0

        cd_norm = float(t.cooldown) / float(self.cooldown_steps) if self.cooldown_steps > 0 else 0.0

        return np.array(wall_norm + tgt_norm + dir_onehot + [cd_norm], dtype=np.float32)





