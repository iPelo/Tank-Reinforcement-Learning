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
    tanks: Dict[AgentId, Tank]
    teams: Dict[AgentId, str]
    walls: Set[Coord]
    steps: int
    phase: int


@dataclass
class LastSeenInfo:
    x: float = 0.0
    y: float = 0.0
    age: int = -1


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    agent_ids: tuple[AgentId, ...]
    teams: Dict[AgentId, str]
    spawn_presets: tuple[Dict[AgentId, Coord], ...]


@dataclass(frozen=True)
class EnvConfig:
    w: int
    h: int
    max_steps: int
    wall_density: float

class TankEnv:
    DEFAULT_ENV_CONFIGS: Dict[str, EnvConfig] = {
        "1v1": EnvConfig(w=21, h=21, max_steps=320, wall_density=0.10),
        "1v2": EnvConfig(w=17, h=17, max_steps=240, wall_density=0.11),
        "2v2": EnvConfig(w=19, h=19, max_steps=260, wall_density=0.10),
    }
    LAYOUTS: Dict[str, LayoutSpec] = {
        "1v1": LayoutSpec(
            name="1v1",
            agent_ids=("player", "enemy"),
            teams={"player": "blue", "enemy": "red"},
            spawn_presets=(
                {"player": (2, 2), "enemy": (12, 12)},
                {"player": (2, 12), "enemy": (12, 2)},
            ),
        ),
        "1v2": LayoutSpec(
            name="1v2",
            agent_ids=("player", "enemy_1", "enemy_2"),
            teams={"player": "blue", "enemy_1": "red", "enemy_2": "red"},
            spawn_presets=(
                {"player": (2, 7), "enemy_1": (12, 3), "enemy_2": (12, 11)},
                {"player": (3, 3), "enemy_1": (11, 3), "enemy_2": (11, 11)},
            ),
        ),
        "2v2": LayoutSpec(
            name="2v2",
            agent_ids=("player", "ally", "enemy_1", "enemy_2"),
            teams={"player": "blue", "ally": "blue", "enemy_1": "red", "enemy_2": "red"},
            spawn_presets=(
                {"player": (2, 4), "ally": (2, 10), "enemy_1": (12, 4), "enemy_2": (12, 10)},
                {"player": (4, 2), "ally": (10, 2), "enemy_1": (4, 12), "enemy_2": (10, 12)},
            ),
        ),
    }

    def __init__(
        self,
        w: int = 15,
        h: int = 15,
        max_steps: int = 200,
        seed: int = 0,
        wall_density: float = 0.12,
        cooldown_steps: int = 5,
        ray_limit:Optional[int] = None,
        layout: str = "1v1",
        spawn_mode: str = "random",
    ) -> None:
        self.w = int(w)
        self.h = int(h)
        self.max_steps = int(max_steps)
        self.wall_density = float(wall_density)
        self.cooldown_steps = int(cooldown_steps)
        self.ray_limit = int(ray_limit) if ray_limit is not None else max(self.w, self.h)
        if layout not in self.LAYOUTS:
            raise ValueError(f"Unknown layout: {layout}")
        if spawn_mode not in {"random", "preset"}:
            raise ValueError(f"Unknown spawn_mode: {spawn_mode}")
        self.layout_name = layout
        self.layout = self.LAYOUTS[layout]
        self.spawn_mode = spawn_mode
        self.agent_ids = self.layout.agent_ids

        self.rng = np.random.default_rng(seed)
        self.state: Optional[EnvState] = None
        self.last_shot: Optional[Dict[str, Tuple[int, int, int, int, bool]]] = None
        self.last_shot_ttl: int = 0
        self.last_seen_enemy: Dict[AgentId, LastSeenInfo] = self._empty_last_seen()
        self.last_step_events: Dict[AgentId, Dict[str, float]] = self._empty_step_events()
        self.visited_cells: Dict[AgentId, Set[Coord]] = self._empty_visited_cells()

    @classmethod
    def default_config(cls, layout: str) -> EnvConfig:
        if layout not in cls.DEFAULT_ENV_CONFIGS:
            raise ValueError(f"Unknown layout: {layout}")
        return cls.DEFAULT_ENV_CONFIGS[layout]

    def _empty_last_seen(self) -> Dict[AgentId, LastSeenInfo]:
        return {agent_id: LastSeenInfo() for agent_id in self.agent_ids}

    def _empty_step_events(self) -> Dict[AgentId, Dict[str, float]]:
        return {
            agent_id: {"took_hit": 0.0, "hit_enemy": 0.0, "heard_shot_fwd": 0.0, "heard_shot_side": 0.0}
            for agent_id in self.agent_ids
        }

    def _empty_visited_cells(self) -> Dict[AgentId, Set[Coord]]:
        return {agent_id: set() for agent_id in self.agent_ids}

    def _tank(self, agent_id: AgentId) -> Tank:
        assert self.state is not None
        return self.state.tanks[agent_id]

    def team_of(self, agent_id: AgentId) -> str:
        if self.state is not None:
            return self.state.teams[agent_id]
        return self.layout.teams[agent_id]

    def team_ids(self) -> tuple[str, ...]:
        if self.state is not None:
            return tuple(dict.fromkeys(self.state.teams.values()))
        return tuple(dict.fromkeys(self.layout.teams.values()))

    def allies_of(self, agent_id: AgentId) -> tuple[AgentId, ...]:
        team_id = self.team_of(agent_id)
        return tuple(candidate for candidate in self.agent_ids if candidate != agent_id and self.team_of(candidate) == team_id)

    def opponents_of(self, agent_id: AgentId) -> tuple[AgentId, ...]:
        team_id = self.team_of(agent_id)
        return tuple(candidate for candidate in self.agent_ids if self.team_of(candidate) != team_id)

    def _opponent_id(self, agent_id: AgentId) -> AgentId:
        opponents = self.opponents_of(agent_id)
        if opponents:
            return opponents[0]
        raise ValueError(f"No opponent defined for {agent_id}")

    def _opponent(self, agent_id: AgentId) -> Tank:
        return self._tank(self._opponent_id(agent_id))

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

    def available_layouts(self) -> tuple[str, ...]:
        return tuple(self.LAYOUTS.keys())

    def _random_spawn_positions(self) -> Dict[AgentId, Coord]:
        positions: Dict[AgentId, Coord] = {}
        taken: Set[Coord] = set()
        for agent_id in self.agent_ids:
            while True:
                candidate = (
                    int(self.rng.integers(1, self.w - 1)),
                    int(self.rng.integers(1, self.h - 1)),
                )
                if candidate in taken:
                    continue
                positions[agent_id] = candidate
                taken.add(candidate)
                break
        return positions

    def _sample_spawn_positions(self) -> Dict[AgentId, Coord]:
        if self.spawn_mode == "preset" and self.layout.spawn_presets:
            idx = int(self.rng.integers(0, len(self.layout.spawn_presets)))
            return dict(self.layout.spawn_presets[idx])
        return self._random_spawn_positions()

    def _spawn_tanks(self, positions: Dict[AgentId, Coord]) -> Dict[AgentId, Tank]:
        tanks: Dict[AgentId, Tank] = {}
        for agent_id in self.agent_ids:
            x, y = positions[agent_id]
            tanks[agent_id] = Tank(
                x=x,
                y=y,
                dir=Direction(int(self.rng.integers(0, 4))),
                cooldown=0,
                team_id=self.layout.teams[agent_id],
            )
        return tanks

    def _spawn_forbidden_cells(self, positions: Dict[AgentId, Coord]) -> Set[Coord]:
        forbidden: Set[Coord] = set()
        for x, y in positions.values():
            forbidden.update(
                {
                    (x, y),
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                }
            )
        return forbidden

    def _all_opponents_reachable(self, tanks: Dict[AgentId, Tank], walls: Set[Coord]) -> bool:
        for agent_id in self.agent_ids:
            start = (tanks[agent_id].x, tanks[agent_id].y)
            for opponent_id in self.opponents_of(agent_id):
                goal = (tanks[opponent_id].x, tanks[opponent_id].y)
                if not self._reachable(start, goal, walls):
                    return False
        return True

    def reset(self, phase: int = 0) -> Dict[AgentId, np.ndarray]:
        phase = int(phase)

        while True:
            positions = self._sample_spawn_positions()
            tanks = self._spawn_tanks(positions)
            forbidden = self._spawn_forbidden_cells(positions)
            walls = generate_walls(self.w, self.h, self.wall_density, self.rng, forbidden=forbidden)

            if not self._all_opponents_reachable(tanks, walls):
                continue

            if any((tank.x, tank.y) in walls for tank in tanks.values()):
                continue

            self.state = EnvState(
                tanks=tanks,
                teams=dict(self.layout.teams),
                walls=walls,
                steps=0,
                phase=phase,
            )
            break

        self.last_shot = None
        self.last_shot_ttl = 0
        self.last_seen_enemy = self._empty_last_seen()
        self.last_step_events = self._empty_step_events()
        self.visited_cells = self._empty_visited_cells()
        for agent_id in self.agent_ids:
            tank = self._tank(agent_id)
            self.visited_cells[agent_id].add((tank.x, tank.y))

        return self.observe_all()

    def step(self, actions: Dict[AgentId, int]) -> Tuple[Dict[AgentId, np.ndarray], Dict[AgentId, float], bool, Dict]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        s = self.state
        s.steps += 1
        pre_positions = {
            agent_id: (self._tank(agent_id).x, self._tank(agent_id).y)
            for agent_id in self.agent_ids
        }
        action_by_agent = {
            agent_id: Action(int(actions.get(agent_id, Action.NOOP)))
            for agent_id in self.agent_ids
        }

        if self.last_shot_ttl > 0:
            self.last_shot_ttl -= 1
            if self.last_shot_ttl == 0:
                self.last_shot = None

        done = False

        for tank in s.tanks.values():
            if tank.cooldown > 0:
                tank.cooldown -= 1

        for agent_id, action in action_by_agent.items():
            self._apply_turn(self._tank(agent_id), action)
        self._apply_moves(action_by_agent)
        post_positions = {
            agent_id: (self._tank(agent_id).x, self._tank(agent_id).y)
            for agent_id in self.agent_ids
        }
        for agent_id, pos in post_positions.items():
            self.visited_cells[agent_id].add(pos)
        enemy_visible = {
            agent_id: bool(self._pick_primary_opponent(agent_id, visible_only=True) is not None)
            for agent_id in self.agent_ids
        }
        pre_distance_to_enemy = {
            agent_id: self._nearest_opponent_distance(agent_id, pre_positions)
            for agent_id in self.agent_ids
        }
        post_distance_to_enemy = {
            agent_id: self._nearest_opponent_distance(agent_id, post_positions)
            for agent_id in self.agent_ids
        }

        current_shots: Dict[str, Tuple[int, int, int, int, bool]] = {}
        hit_targets: Set[AgentId] = set()
        shooter_hits = {agent_id: False for agent_id in self.agent_ids}
        if s.phase >= 1:
            hit_targets, shooter_hits = self._resolve_shots(current_shots, action_by_agent)

        for target_id in hit_targets:
            self._tank(target_id).alive = False

        team_alive_counts = self._team_alive_counts()
        surviving_teams = [team_id for team_id, count in team_alive_counts.items() if count > 0]
        winning_team = surviving_teams[0] if len(surviving_teams) == 1 else None
        if len(surviving_teams) <= 1:
            done = True

        if s.steps >= self.max_steps:
            done = True

        draw = bool(done and winning_team is None)
        agent_hits = {
            agent_id: bool(shooter_hits[agent_id])
            for agent_id in self.agent_ids
        }
        agent_alive = {
            agent_id: self._tank(agent_id).alive
            for agent_id in self.agent_ids
        }
        agent_wins = {
            agent_id: bool(winning_team is not None and self.team_of(agent_id) == winning_team)
            for agent_id in self.agent_ids
        }
        team_wins = {
            team_id: bool(team_id == winning_team)
            for team_id in self.team_ids()
        }
        rewards = {
            agent_id: self._reward_for_side(
                action=action_by_agent[agent_id],
                win=agent_wins[agent_id],
                loss=bool(winning_team is not None and self.team_of(agent_id) != winning_team),
                draw=draw,
                phase=s.phase,
                hit_enemy=shooter_hits[agent_id],
                took_hit=agent_id in hit_targets,
                enemy_visible=enemy_visible[agent_id],
                pre_enemy_distance=pre_distance_to_enemy[agent_id],
                post_enemy_distance=post_distance_to_enemy[agent_id],
            )
            for agent_id in self.agent_ids
        }

        player_win = agent_wins.get("player", False)
        enemy_win = agent_wins.get("enemy", False)
        player_hit = bool("player" in hit_targets)
        enemy_hit = bool("enemy" in hit_targets)

        info = StepInfo(
            player_win=player_win,
            enemy_win=enemy_win,
            draw=draw,
            player_hit=player_hit,
            enemy_hit=enemy_hit,
            steps=s.steps,
            phase=s.phase,
            agent_wins=agent_wins,
            agent_hits=agent_hits,
            agent_alive=agent_alive,
            team_wins=team_wins,
            team_alive_counts=team_alive_counts,
            winning_team=winning_team,
        )
        if current_shots:
            self.last_shot = current_shots
            self.last_shot_ttl = 6

        self._update_memory(hit_targets=hit_targets, shooter_hits=shooter_hits, current_shots=current_shots)

        team_rewards: Dict[str, float] = {}
        for agent_id, reward in rewards.items():
            team_id = self.team_of(agent_id)
            team_rewards[team_id] = team_rewards.get(team_id, 0.0) + float(reward)

        obs_by_agent = self.observe_all()
        return obs_by_agent, rewards, bool(done), {
            "info": info,
            "shot": self.last_shot,
            "shot_ttl": self.last_shot_ttl,
            "agent_ids": self.agent_ids,
            "teams": dict(s.teams),
            "agent_rewards": dict(rewards),
            "team_rewards": team_rewards,
            "agent_done": {agent_id: bool(done) for agent_id in self.agent_ids},
            "team_done": {team_id: bool(done) for team_id in self.team_ids()},
            "team_obs": self.team_observe_all(),
            "actions": action_by_agent,
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

    def _apply_moves(self, action_by_agent: Dict[AgentId, Action]) -> None:
        assert self.state is not None
        current_positions = {
            agent_id: (self._tank(agent_id).x, self._tank(agent_id).y)
            for agent_id in self.agent_ids
        }
        desired_positions = dict(current_positions)

        for agent_id, action in action_by_agent.items():
            tank = self._tank(agent_id)
            if not tank.alive:
                continue
            if action == Action.FWD:
                desired_positions[agent_id] = self._move_target(tank, True)
            elif action == Action.BWD:
                desired_positions[agent_id] = self._move_target(tank, False)
            if self._is_wall(*desired_positions[agent_id]):
                desired_positions[agent_id] = current_positions[agent_id]

        for agent_id in self.agent_ids:
            for other_id in self.agent_ids:
                if agent_id >= other_id:
                    continue
                if desired_positions[agent_id] == desired_positions[other_id]:
                    desired_positions[agent_id] = current_positions[agent_id]
                    desired_positions[other_id] = current_positions[other_id]
                elif (
                    desired_positions[agent_id] == current_positions[other_id]
                    and desired_positions[other_id] == current_positions[agent_id]
                ):
                    desired_positions[agent_id] = current_positions[agent_id]
                    desired_positions[other_id] = current_positions[other_id]

        for agent_id, (next_x, next_y) in desired_positions.items():
            tank = self._tank(agent_id)
            tank.x, tank.y = next_x, next_y

    def _trace_shot(self, shooter_id: AgentId) -> tuple[int, int, int, int, bool, AgentId | None]:
        shooter = self._tank(shooter_id)
        fwd = dir_to_vec(shooter.dir)
        x0, y0 = shooter.x, shooter.y
        dx, dy = fwd

        hit = False
        target_id: AgentId | None = None
        endx, endy = x0, y0
        x, y = x0, y0
        target_positions = {
            candidate: (self._tank(candidate).x, self._tank(candidate).y)
            for candidate in self.opponents_of(shooter_id)
            if self._tank(candidate).alive
        }

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

            for candidate_id, position in target_positions.items():
                if (x, y) == position:
                    hit = True
                    target_id = candidate_id
                    break
            if hit:
                break

        return x0, y0, endx, endy, hit, target_id

    def _resolve_shots(
        self,
        current_shots: Dict[str, Tuple[int, int, int, int, bool]],
        action_by_agent: Dict[AgentId, Action],
    ) -> tuple[Set[AgentId], Dict[AgentId, bool]]:
        hit_targets: Set[AgentId] = set()
        shooter_hits = {agent_id: False for agent_id in self.agent_ids}
        for agent_id, action in action_by_agent.items():
            shooter = self._tank(agent_id)
            if action != Action.SHOOT or shooter.cooldown != 0 or not shooter.alive:
                continue
            shot = self._trace_shot(agent_id)
            current_shots[agent_id] = shot[:5]
            shooter.cooldown = self.cooldown_steps
            if shot[5] is not None:
                hit_targets.add(shot[5])
                shooter_hits[agent_id] = True
        return hit_targets, shooter_hits

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

    def _nearest_opponent_distance(
        self,
        agent_id: AgentId,
        positions: Dict[AgentId, Coord],
    ) -> int | None:
        origin = positions[agent_id]
        distances = [
            abs(origin[0] - positions[opponent_id][0]) + abs(origin[1] - positions[opponent_id][1])
            for opponent_id in self.opponents_of(agent_id)
        ]
        if not distances:
            return None
        return min(distances)

    def _reward_for_side(
        self,
        action: Action,
        win: bool,
        loss: bool,
        draw: bool,
        phase: int,
        hit_enemy: bool,
        took_hit: bool,
        enemy_visible: bool,
        pre_enemy_distance: int | None,
        post_enemy_distance: int | None,
    ) -> float:
        if win:
            return 1.0
        if loss:
            return -1.0
        if draw:
            return 0.0

        reward = -0.006

        if enemy_visible:
            reward += 0.01

        if hit_enemy:
            reward += 0.22
        if took_hit:
            reward -= 0.22

        if (
            pre_enemy_distance is not None
            and post_enemy_distance is not None
            and action in (Action.FWD, Action.BWD)
        ):
            if post_enemy_distance < pre_enemy_distance:
                reward += 0.012
            elif post_enemy_distance > pre_enemy_distance:
                reward -= 0.008

        if action == Action.SHOOT and phase >= 1:
            if hit_enemy:
                reward += 0.02
            elif enemy_visible:
                reward -= 0.012
            else:
                reward -= 0.03
        elif action == Action.NOOP and not enemy_visible:
            reward -= 0.004

        return reward

    def _relative_direction_features(self, observer: Tank, target_xy: Coord) -> tuple[float, float]:
        tx, ty = target_xy
        dx = tx - observer.x
        dy = ty - observer.y
        fwd_vec = dir_to_vec(observer.dir)
        right_vec = dir_to_vec(turn_right(observer.dir))
        fwd_component = float(dx * fwd_vec[0] + dy * fwd_vec[1])
        side_component = float(dx * right_vec[0] + dy * right_vec[1])
        return fwd_component, side_component

    def _pick_primary_opponent(self, agent_id: AgentId, visible_only: bool = False) -> Tank | None:
        observer = self._tank(agent_id)
        candidates: list[Tank] = []
        for opponent_id in self.opponents_of(agent_id):
            opponent = self._tank(opponent_id)
            if not opponent.alive:
                continue
            if visible_only and not self._is_visible(observer, opponent):
                continue
            candidates.append(opponent)
        if not candidates:
            return None
        return min(candidates, key=lambda tank: abs(tank.x - observer.x) + abs(tank.y - observer.y))

    def _update_memory(
        self,
        hit_targets: Set[AgentId],
        shooter_hits: Dict[AgentId, bool],
        current_shots: Dict[str, Tuple[int, int, int, int, bool]],
    ) -> None:
        self.last_step_events = self._empty_step_events()
        for agent_id in self.agent_ids:
            self.last_step_events[agent_id]["took_hit"] = float(agent_id in hit_targets)
            self.last_step_events[agent_id]["hit_enemy"] = float(shooter_hits.get(agent_id, False))

            observer = self._tank(agent_id)
            target = self._pick_primary_opponent(agent_id, visible_only=True)
            if target is not None:
                self.last_seen_enemy[agent_id] = LastSeenInfo(
                    x=target.x / max(1, self.w - 1),
                    y=target.y / max(1, self.h - 1),
                    age=0,
                )
            elif self.last_seen_enemy[agent_id].age >= 0:
                self.last_seen_enemy[agent_id].age += 1

            hearing_target: Tank | None = None
            for shooter_id in current_shots:
                if self.team_of(shooter_id) != self.team_of(agent_id):
                    hearing_target = self._tank(shooter_id)
                    break
            if hearing_target is not None:
                fwd, side = self._relative_direction_features(observer, (hearing_target.x, hearing_target.y))
                self.last_step_events[agent_id]["heard_shot_fwd"] = 1.0 if fwd >= 0 else -1.0
                self.last_step_events[agent_id]["heard_shot_side"] = 1.0 if side >= 0 else -1.0

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

    def _raycast_enemy_dist(self, origin: Coord, direction: Coord, limit: int, enemies: tuple[Tank, ...]) -> int:
        ox, oy = origin
        dx, dy = direction
        dist = 0
        x, y = ox + dx, oy + dy

        while dist < limit:
            if self._is_wall(x,y):
                return -1
            for enemy in enemies:
                if (x,y) == (enemy.x, enemy.y):
                    return dist
            dist += 1
            x += dx
            y += dy

        return -1

    def observe(self, agent_id: AgentId) -> np.ndarray:
        assert self.state is not None
        if agent_id not in self.state.tanks:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        t = self._tank(agent_id)
        enemies = tuple(self._tank(opponent_id) for opponent_id in self.opponents_of(agent_id) if self._tank(opponent_id).alive)

        fwd = dir_to_vec(t.dir)
        back = (-fwd[0], -fwd[1])
        left = dir_to_vec(turn_left(t.dir))
        right = dir_to_vec(turn_right(t.dir))
        dirs = [fwd, back, left, right]

        limit =  self.ray_limit
        origin = (t.x, t.y)

        wall_d = [self._raycast_wall_dist(origin, d, limit) for d in dirs]
        enemy_visible = any(self._is_visible(t, enemy) for enemy in enemies)
        if enemy_visible:
            enemy_d = [self._raycast_enemy_dist(origin, d, limit, enemies) for d in dirs]
        else:
            enemy_d = [-1, -1, -1, -1]

        wall_norm = [d / limit for d in wall_d]
        enemy_norm = [(-1.0 if d < 0 else d / limit) for d in enemy_d]

        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_onehot[int(t.dir)] = 1.0

        cd_norm = float(t.cooldown) / float(self.cooldown_steps) if self.cooldown_steps > 0 else 0.0
        enemy_has_shot = 0.0
        for enemy in enemies:
            if self._is_visible(enemy, t):
                if enemy.x == t.x:
                    goal_dir = Direction.S if t.y > enemy.y else Direction.N
                else:
                    goal_dir = Direction.E if t.x > enemy.x else Direction.W
                if enemy.dir == goal_dir and enemy.cooldown == 0:
                    enemy_has_shot = 1.0
                    break

        heard_enemy_shot = 0.0
        if self.last_shot is not None and self.last_shot_ttl > 0:
            heard_enemy_shot = 1.0 if any(self.team_of(shooter_id) != self.team_of(agent_id) for shooter_id in self.last_shot) else 0.0

        pos_norm = [t.x / max(1, self.w - 1), t.y / max(1, self.h - 1)]
        step_norm = float(self.state.steps) / float(self.max_steps) if self.max_steps > 0 else 0.0
        last_seen = self.last_seen_enemy[agent_id]
        if last_seen.age >= 0:
            last_seen_xy = (
                last_seen.x * max(1, self.w - 1),
                last_seen.y * max(1, self.h - 1),
            )
            last_seen_fwd, last_seen_side = self._relative_direction_features(t, last_seen_xy)
            last_seen_fwd /= max(1, self.h - 1)
            last_seen_side /= max(1, self.w - 1)
            last_seen_age = (
                min(float(last_seen.age) / float(self.max_steps), 1.0)
                if self.max_steps > 0
                else 1.0
            )
            last_seen_valid = 1.0
            search_pressure = max(0.0, 1.0 - last_seen_age)
        else:
            last_seen_fwd = 0.0
            last_seen_side = 0.0
            last_seen_age = 1.0
            last_seen_valid = 0.0
            search_pressure = 0.0
        walkable_cells = (self.w * self.h) - len(self.state.walls)
        explored_ratio = len(self.visited_cells[agent_id]) / max(1, walkable_cells)
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
                float(len(self.allies_of(agent_id))),
                float(len(self.opponents_of(agent_id))),
                step_norm,
                last_seen_fwd,
                last_seen_side,
                last_seen_age,
                last_seen_valid,
                search_pressure,
                float(explored_ratio),
                recent_events["took_hit"],
                recent_events["hit_enemy"],
                recent_events["heard_shot_fwd"],
                recent_events["heard_shot_side"],
            ],
            dtype=np.float32,
        )

    def observe_all(self) -> Dict[AgentId, np.ndarray]:
        return {agent_id: self.observe(agent_id) for agent_id in self.agent_ids}

    def team_observe_all(self) -> Dict[str, Dict[AgentId, np.ndarray]]:
        team_obs: Dict[str, Dict[AgentId, np.ndarray]] = {}
        for agent_id in self.agent_ids:
            team_id = self.team_of(agent_id)
            team_obs.setdefault(team_id, {})[agent_id] = self.observe(agent_id)
        return team_obs

    def _team_alive_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {team_id: 0 for team_id in self.team_ids()}
        for agent_id in self.agent_ids:
            if self._tank(agent_id).alive:
                counts[self.team_of(agent_id)] += 1
        return counts
