from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import numpy as np
from .entities import Action, Direction, StepInfo, Tank, dir_to_vec, turn_left, turn_right
from .map_gen import generate_walls

Coord = Tuple[int, int]
AgentId = str

@dataclass
class EnvState:
    player: Tank
    enemy: Tank
    walls: Set[Coord]
    steps: int
    phase: int


@dataclass
class LastSeenInfo:
    dx: float = 0.0
    dy: float = 0.0
    age: int = -1

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
        self.last_seen_enemy: Dict[AgentId, LastSeenInfo] = {
            "player": LastSeenInfo(),
            "enemy": LastSeenInfo(),
        }
        self.last_step_events: Dict[AgentId, Dict[str, float]] = {
            "player": {"took_hit": 0.0, "hit_enemy": 0.0, "heard_shot_fwd": 0.0, "heard_shot_side": 0.0},
            "enemy": {"took_hit": 0.0, "hit_enemy": 0.0, "heard_shot_fwd": 0.0, "heard_shot_side": 0.0},
        }

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

    def reset(self, phase: int = 0) -> Dict[AgentId, np.ndarray]:
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
        self.last_seen_enemy = {
            "player": LastSeenInfo(),
            "enemy": LastSeenInfo(),
        }
        self.last_step_events = {
            "player": {"took_hit": 0.0, "hit_enemy": 0.0, "heard_shot_fwd": 0.0, "heard_shot_side": 0.0},
            "enemy": {"took_hit": 0.0, "hit_enemy": 0.0, "heard_shot_fwd": 0.0, "heard_shot_side": 0.0},
        }

        return self.observe_all()

    def step(self, actions: Dict[AgentId, int]) -> Tuple[Dict[AgentId, np.ndarray], Dict[AgentId, float], bool, Dict]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        s = self.state
        s.steps += 1
        player_action = Action(int(actions["player"]))
        enemy_action = Action(int(actions["enemy"]))

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

        rewards = {
            "player": self._reward_for_side(
                action=player_action,
                win=player_win,
                loss=enemy_win,
                draw=draw,
                phase=s.phase,
            ),
            "enemy": self._reward_for_side(
                action=enemy_action,
                win=enemy_win,
                loss=player_win,
                draw=draw,
                phase=s.phase,
            ),
        }

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

        self._update_memory(player_hit=player_hit, enemy_hit=enemy_hit, current_shots=current_shots)

        return self.observe_all(), rewards, bool(done), {
            "info": info,
            "shot": self.last_shot,
            "shot_ttl": self.last_shot_ttl,
            "actions": {
                "player": player_action,
                "enemy": enemy_action,
            },
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

    def _is_visible(self, observer: Tank, target: Tank) -> bool:
        same_row = observer.y == target.y
        same_col = observer.x == target.x
        if not same_row and not same_col:
            return False
        return self._clear_line((observer.x, observer.y), (target.x, target.y))

    def _best_turn_toward(self, tank: Tank, goal_dir: Direction) -> Action:
        left_steps = (int(tank.dir) - int(goal_dir)) % 4
        right_steps = (int(goal_dir) - int(tank.dir)) % 4
        return Action.LEFT if left_steps <= right_steps else Action.RIGHT

    def _reward_for_side(self, action: Action, win: bool, loss: bool, draw: bool, phase: int) -> float:
        if win:
            return 1.0
        if loss:
            return -1.0
        if draw:
            return 0.0
        if action == Action.SHOOT and phase >= 1:
            return -0.02
        return -0.01

    def _relative_direction_features(self, observer: Tank, target_xy: Coord) -> tuple[float, float]:
        tx, ty = target_xy
        dx = tx - observer.x
        dy = ty - observer.y
        fwd_vec = dir_to_vec(observer.dir)
        right_vec = dir_to_vec(turn_right(observer.dir))
        fwd_component = float(dx * fwd_vec[0] + dy * fwd_vec[1])
        side_component = float(dx * right_vec[0] + dy * right_vec[1])
        return fwd_component, side_component

    def _update_memory(
        self,
        player_hit: bool,
        enemy_hit: bool,
        current_shots: Dict[str, Tuple[int, int, int, int, bool]],
    ) -> None:
        assert self.state is not None
        player = self.state.player
        enemy = self.state.enemy

        self.last_step_events = {
            "player": {
                "took_hit": float(player_hit),
                "hit_enemy": float(enemy_hit),
                "heard_shot_fwd": 0.0,
                "heard_shot_side": 0.0,
            },
            "enemy": {
                "took_hit": float(enemy_hit),
                "hit_enemy": float(player_hit),
                "heard_shot_fwd": 0.0,
                "heard_shot_side": 0.0,
            },
        }

        for agent_id, observer, target in (
            ("player", player, enemy),
            ("enemy", enemy, player),
        ):
            if self._is_visible(observer, target):
                self.last_seen_enemy[agent_id] = LastSeenInfo(
                    dx=(target.x - observer.x) / max(1, self.w - 1),
                    dy=(target.y - observer.y) / max(1, self.h - 1),
                    age=0,
                )
            elif self.last_seen_enemy[agent_id].age >= 0:
                self.last_seen_enemy[agent_id].age += 1

        # Encode only coarse directional audio cues so the agent must still remember context.
        if "enemy" in current_shots:
            fwd, side = self._relative_direction_features(player, (enemy.x, enemy.y))
            self.last_step_events["player"]["heard_shot_fwd"] = 1.0 if fwd >= 0 else -1.0
            self.last_step_events["player"]["heard_shot_side"] = 1.0 if side >= 0 else -1.0
        if "player" in current_shots:
            fwd, side = self._relative_direction_features(enemy, (player.x, player.y))
            self.last_step_events["enemy"]["heard_shot_fwd"] = 1.0 if fwd >= 0 else -1.0
            self.last_step_events["enemy"]["heard_shot_side"] = 1.0 if side >= 0 else -1.0

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

    def observe(self, agent_id: AgentId) -> np.ndarray:
        assert self.state is not None
        if agent_id == "player":
            t = self.state.player
            enemy = self.state.enemy
        elif agent_id == "enemy":
            t = self.state.enemy
            enemy = self.state.player
        else:
            raise ValueError(f"Unknown agent_id: {agent_id}")

        fwd = dir_to_vec(t.dir)
        back = (-fwd[0], -fwd[1])
        left = dir_to_vec(turn_left(t.dir))
        right = dir_to_vec(turn_right(t.dir))
        dirs = [fwd, back, left, right]

        limit =  self.ray_limit
        origin = (t.x, t.y)

        wall_d = [self._raycast_wall_dist(origin, d, limit) for d in dirs]
        enemy_visible = self._is_visible(t, enemy)
        if enemy_visible:
            enemy_d = [self._raycast_enemy_dist(origin, d, limit, enemy) for d in dirs]
        else:
            enemy_d = [-1, -1, -1, -1]

        wall_norm = [d / limit for d in wall_d]
        enemy_norm = [(-1.0 if d < 0 else d / limit) for d in enemy_d]

        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_onehot[int(t.dir)] = 1.0

        cd_norm = float(t.cooldown) / float(self.cooldown_steps) if self.cooldown_steps > 0 else 0.0
        enemy_has_shot = 0.0
        if enemy_visible and self._clear_line((enemy.x, enemy.y), (t.x, t.y)):
            if enemy.x == t.x:
                goal_dir = Direction.S if t.y > enemy.y else Direction.N
            else:
                goal_dir = Direction.E if t.x > enemy.x else Direction.W
            enemy_has_shot = 1.0 if enemy.dir == goal_dir and enemy.cooldown == 0 else 0.0

        heard_enemy_shot = 0.0
        if self.last_shot is not None and self.last_shot_ttl > 0:
            shot_key = "enemy" if agent_id == "player" else "player"
            heard_enemy_shot = 1.0 if shot_key in self.last_shot else 0.0

        pos_norm = [t.x / max(1, self.w - 1), t.y / max(1, self.h - 1)]
        step_norm = float(self.state.steps) / float(self.max_steps) if self.max_steps > 0 else 0.0
        last_seen = self.last_seen_enemy[agent_id]
        last_seen_age = (
            min(float(last_seen.age) / float(self.max_steps), 1.0)
            if last_seen.age >= 0 and self.max_steps > 0
            else -1.0
        )
        recent_events = self.last_step_events[agent_id]

        return np.array(
            wall_norm
            + enemy_norm
            + dir_onehot
            + pos_norm
            + [
                cd_norm,
                float(enemy_visible),
                enemy_has_shot,
                heard_enemy_shot,
                step_norm,
                last_seen.dx,
                last_seen.dy,
                last_seen_age,
                recent_events["took_hit"],
                recent_events["hit_enemy"],
                recent_events["heard_shot_fwd"],
                recent_events["heard_shot_side"],
            ],
            dtype=np.float32,
        )

    def observe_all(self) -> Dict[AgentId, np.ndarray]:
        return {
            "player": self.observe("player"),
            "enemy": self.observe("enemy"),
        }
