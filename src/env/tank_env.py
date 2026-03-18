from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import numpy as np
from .entities import Action, Direction, StepInfo, Tank, dir_to_vec, turn_left, turn_right
from .map_gen import generate_walls

Coord = Tuple[int, int]

@dataclass
class EnvState:
    player: Tank
    enemy: Tank
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
        self.last_shot: Optional[Dict[str, Tuple[int, int, int, int, bool]]] = None
        self.last_shot_ttl: int = 0

    def _reachable(self, start: Coord, goal: Coord,walls: Set[Coord]) -> bool:
        from collections import deque

        q = deque([start])
        seen = {start}

        while q:
            x,y = q.popleft()
            if (x,y) == goal:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not(0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                if(nx,ny) in walls:
                    continue
                if(nx,ny) in seen:
                    continue
                seen.add((nx,ny))
                q.append((nx,ny))
        return  False

    def reset(self, phase: int = 0) -> np.ndarray:
        phase = int(phase)

        while True:
            player = Tank(
                x=int(self.rng.integers(1, self.w - 1)),
                y=int(self.rng.integers(1, self.h - 1)),
                dir=Direction(int(self.rng.integers(0,4))),
                cooldown=0,
            )
            enemy = Tank(
                x=int(self.rng.integers(1, self.w - 1)),
                y=int(self.rng.integers(1, self.h - 1)),
                dir=Direction(int(self.rng.integers(0,4))),
                cooldown=0,
            )

            if (player.x, player.y) == (enemy.x, enemy.y):
                continue

            forbidden = {
                (player.x, player.y),
                (enemy.x, enemy.y),
                (player.x + 1, player.y),
                (player.x - 1, player.y),
                (player.x, player.y + 1),
                (player.x, player.y - 1),
                (enemy.x + 1, enemy.y),
                (enemy.x - 1, enemy.y),
                (enemy.x, enemy.y + 1),
                (enemy.x, enemy.y - 1),
            }

            walls = generate_walls(self.w, self.h, self.wall_density, self.rng, forbidden=forbidden)

            if not self._reachable((player.x, player.y), (enemy.x, enemy.y), walls):
                continue

            if (player.x, player.y) in walls or (enemy.x, enemy.y) in walls:
                continue

            self.state = EnvState(player=player, enemy=enemy, walls=walls, steps=0, phase=phase)
            break

        self.last_shot = None
        self.last_shot_ttl = 0

        return self._get_obs()

    def step(self,action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        s = self.state
        s.steps += 1
        player_action = Action(int(action))
        enemy_action = self._enemy_policy()

        if self.last_shot_ttl > 0:
            self.last_shot_ttl -= 1
            if self.last_shot_ttl == 0:
                self.last_shot = None

        current_shots: Dict[str, Tuple[int, int, int, int, bool]] = {}
        player_hit = False
        enemy_hit = False
        done = False

        for tank in (s.player, s.enemy):
            if tank.cooldown > 0:
                tank.cooldown -= 1

        self._apply_turn(s.player, player_action)
        self._apply_turn(s.enemy, enemy_action)
        self._apply_moves(player_action, enemy_action)

        if s.phase >= 1:
            player_hit = self._resolve_shot(current_shots, "player", s.player, s.enemy, player_action)
            enemy_hit = self._resolve_shot(current_shots, "enemy", s.enemy, s.player, enemy_action)

        if enemy_hit:
            s.enemy.alive = False
        if player_hit:
            s.player.alive = False

        player_win = enemy_hit and not player_hit
        enemy_win = player_hit and not enemy_hit
        draw = player_hit and enemy_hit

        if not s.player.alive or not s.enemy.alive:
            done = True

        if s.steps >= self.max_steps:
            done = True

        if player_win:
            reward = 1.0
        elif enemy_win:
            reward = -1.0
        elif draw:
            reward = 0.0
        elif player_action == Action.SHOOT and s.phase >= 1:
            reward = -0.02
        else:
            reward = -0.01

        info = StepInfo(
            player_win=player_win,
            enemy_win=enemy_win,
            draw=draw,
            player_hit=player_hit,
            enemy_hit=enemy_hit,
            steps=s.steps,
            phase=s.phase,
        )
        if current_shots:
            self.last_shot = current_shots
            self.last_shot_ttl = 6

        return self._get_obs(), float(reward), bool(done), {
            "info": info,
            "shot": self.last_shot,
            "shot_ttl": self.last_shot_ttl,
            "enemy_action": enemy_action,
        }

    def _in_bounds(self, x:int,y:int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def _is_wall(self, x: int, y: int) -> bool:
        if not self._in_bounds(x,y):
            return True
        assert self.state is not None
        return (x,y) in self.state.walls

    def _apply_turn(self, tank: Tank, action: Action) -> None:
        if action == Action.LEFT:
            tank.dir = turn_left(tank.dir)
        elif action == Action.RIGHT:
            tank.dir = turn_right(tank.dir)

    def _move_target(self, tank: Tank, forward: bool) -> Coord:
        dx, dy = dir_to_vec(tank.dir)
        if not forward:
            dx, dy = -dx, -dy
        return tank.x + dx, tank.y + dy

    def _apply_moves(self, player_action: Action, enemy_action: Action) -> None:
        assert self.state is not None

        player_next = (self.state.player.x, self.state.player.y)
        enemy_next = (self.state.enemy.x, self.state.enemy.y)

        if player_action == Action.FWD:
            player_next = self._move_target(self.state.player, True)
        elif player_action == Action.BWD:
            player_next = self._move_target(self.state.player, False)

        if enemy_action == Action.FWD:
            enemy_next = self._move_target(self.state.enemy, True)
        elif enemy_action == Action.BWD:
            enemy_next = self._move_target(self.state.enemy, False)

        if self._is_wall(*player_next):
            player_next = (self.state.player.x, self.state.player.y)
        if self._is_wall(*enemy_next):
            enemy_next = (self.state.enemy.x, self.state.enemy.y)

        player_cur = (self.state.player.x, self.state.player.y)
        enemy_cur = (self.state.enemy.x, self.state.enemy.y)

        same_cell = player_next == enemy_next
        swap = player_next == enemy_cur and enemy_next == player_cur
        if same_cell or swap:
            player_next = player_cur
            enemy_next = enemy_cur

        self.state.player.x, self.state.player.y = player_next
        self.state.enemy.x, self.state.enemy.y = enemy_next

    def _trace_shot(self, shooter: Tank, defender: Tank) -> Tuple[int, int, int, int, bool]:
        fwd = dir_to_vec(shooter.dir)
        x0, y0 = shooter.x, shooter.y
        dx, dy = fwd

        hit = False
        endx, endy = x0, y0
        x, y = x0, y0

        for _ in range(self.ray_limit):
            nx, ny = x + dx, y + dy

            if not self._in_bounds(nx, ny):
                endx, endy = x, y
                break

            if self._is_wall(nx, ny):
                endx, endy = nx, ny
                break

            x, y = nx, ny
            endx, endy = x, y

            if (x, y) == (defender.x, defender.y):
                hit = True
                break

        return x0, y0, endx, endy, hit

    def _resolve_shot(
        self,
        current_shots: Dict[str, Tuple[int, int, int, int, bool]],
        shot_key: str,
        shooter: Tank,
        defender: Tank,
        action: Action,
    ) -> bool:
        if action != Action.SHOOT or shooter.cooldown != 0 or not shooter.alive or not defender.alive:
            return False

        shot = self._trace_shot(shooter, defender)
        current_shots[shot_key] = shot
        shooter.cooldown = self.cooldown_steps
        return bool(shot[-1])

    def _clear_line(self, origin: Coord, goal: Coord) -> bool:
        ox, oy = origin
        gx, gy = goal

        if ox == gx:
            step = 1 if gy > oy else -1
            for y in range(oy + step, gy, step):
                if self._is_wall(ox, y):
                    return False
            return True

        if oy == gy:
            step = 1 if gx > ox else -1
            for x in range(ox + step, gx, step):
                if self._is_wall(x, oy):
                    return False
            return True

        return False

    def _best_turn_toward(self, tank: Tank, goal_dir: Direction) -> Action:
        left_steps = (int(tank.dir) - int(goal_dir)) % 4
        right_steps = (int(goal_dir) - int(tank.dir)) % 4
        return Action.LEFT if left_steps <= right_steps else Action.RIGHT

    def _enemy_policy(self) -> Action:
        assert self.state is not None
        enemy = self.state.enemy
        player = self.state.player

        if self.state.phase == 0:
            return Action.NOOP

        if enemy.cooldown == 0 and self._clear_line((enemy.x, enemy.y), (player.x, player.y)):
            if enemy.x == player.x:
                goal_dir = Direction.S if player.y > enemy.y else Direction.N
            else:
                goal_dir = Direction.E if player.x > enemy.x else Direction.W
            if enemy.dir == goal_dir:
                return Action.SHOOT
            return self._best_turn_toward(enemy, goal_dir)

        if self.state.phase == 1:
            return Action.NOOP

        dx = player.x - enemy.x
        dy = player.y - enemy.y

        if abs(dx) >= abs(dy):
            goal_dir = Direction.E if dx > 0 else Direction.W
        else:
            goal_dir = Direction.S if dy > 0 else Direction.N

        if enemy.dir != goal_dir:
            return self._best_turn_toward(enemy, goal_dir)

        nx, ny = self._move_target(enemy, True)
        if (nx, ny) != (player.x, player.y) and not self._is_wall(nx, ny):
            return Action.FWD

        return Action.LEFT if self.rng.random() < 0.5 else Action.RIGHT

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

    def _raycast_enemy_dist(self, origin: Coord, direction: Coord, limit: int, enemy: Tank) -> int:
        ox, oy = origin
        dx, dy = direction
        dist = 0
        x, y = ox + dx, oy + dy

        while dist < limit:
            if self._is_wall(x,y):
                return -1
            if (x,y) == (enemy.x, enemy.y):
                return dist
            dist += 1
            x += dx
            y += dy

        return -1

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        t = self.state.player
        enemy = self.state.enemy

        fwd = dir_to_vec(t.dir)
        back = (-fwd[0], -fwd[1])
        left = dir_to_vec(turn_left(t.dir))
        right = dir_to_vec(turn_right(t.dir))
        dirs = [fwd, back, left, right]

        limit =  self.ray_limit
        origin = (t.x, t.y)

        wall_d = [self._raycast_wall_dist(origin, d, limit) for d in dirs]
        enemy_d = [self._raycast_enemy_dist(origin, d, limit, enemy) for d in dirs]

        wall_norm = [d / limit for d in wall_d]
        enemy_norm = [(-1.0 if d < 0 else d / limit) for d in enemy_d]

        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_onehot[int(t.dir)] = 1.0

        cd_norm = float(t.cooldown) / float(self.cooldown_steps) if self.cooldown_steps > 0 else 0.0
        enemy_visible = 1.0 if any(d >= 0 for d in enemy_d) else 0.0
        enemy_has_shot = 0.0
        if self._clear_line((enemy.x, enemy.y), (t.x, t.y)):
            if enemy.x == t.x:
                goal_dir = Direction.S if t.y > enemy.y else Direction.N
            else:
                goal_dir = Direction.E if t.x > enemy.x else Direction.W
            enemy_has_shot = 1.0 if enemy.dir == goal_dir and enemy.cooldown == 0 else 0.0

        return np.array(
            wall_norm + enemy_norm + dir_onehot + [cd_norm, enemy_visible, enemy_has_shot],
            dtype=np.float32,
        )



