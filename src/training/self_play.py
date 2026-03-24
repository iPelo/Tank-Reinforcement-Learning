import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.agents.policy import ActorCritic, HiddenState, PolicyModel, RecurrentActorCritic
from src.evaluation.checkpoint_match import load_policy, run_match_series
from src.env.tank_env import TankEnv
from src.training.buffer import RecurrentRolloutBuffer, RolloutBuffer
from src.training.ppo import PPO, PPOConfig

CHECKPOINT_VERSION = 3


def build_ckpt_payload(
    model: PolicyModel,
    obs_dim: int,
    act_dim: int,
    updates: int,
    phase: int,
    agent_ids: tuple[str, ...],
    layout: str,
    cfg: PPOConfig | None = None,
) -> dict[str, Any]:
    policy_type = "recurrent_shared_policy" if isinstance(model, RecurrentActorCritic) else "shared_policy"
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "training_mode": "self_play",
        "policy_type": policy_type,
        "agent_ids": list(agent_ids),
        "layout": layout,
        "model": model.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "updates": updates,
        "phase": phase,
        "hidden": int(getattr(model, "hidden", 128)),
    }
    if isinstance(model, RecurrentActorCritic):
        payload["recurrent_hidden"] = model.recurrent_hidden
    if cfg is not None:
        payload["ppo_config"] = asdict(cfg)
    return payload


def save_ckpt(
    path: Path,
    model: PolicyModel,
    obs_dim: int,
    act_dim: int,
    updates: int,
    phase: int,
    agent_ids: tuple[str, ...],
    layout: str,
    cfg: PPOConfig | None = None,
) -> None:
    torch.save(
        build_ckpt_payload(
            model=model,
            obs_dim=obs_dim,
            act_dim=act_dim,
            updates=updates,
            phase=phase,
            agent_ids=agent_ids,
            layout=layout,
            cfg=cfg,
        ),
        path,
    )


def load_ckpt(path: Path, model: PolicyModel, device: torch.device) -> tuple[int, int, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return int(ckpt.get("updates", 0)), int(ckpt.get("phase", 0)), ckpt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Starting phase. Default is 0 so training follows the curriculum from the beginning.",
    )
    ap.add_argument(
        "--single-phase",
        action="store_true",
        help="Disable automatic phase promotion and train only the selected phase.",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume training from.",
    )
    ap.add_argument(
        "--min-updates-per-phase",
        type=int,
        default=10,
        help="Minimum PPO updates before phase promotion is allowed.",
    )
    ap.add_argument(
        "--advance-win-rate",
        type=float,
        default=0.70,
        help="Rolling win-rate threshold used to unlock the next phase.",
    )
    ap.add_argument(
        "--opponent-checkpoint",
        type=str,
        default="",
        help="Optional frozen opponent checkpoint. When set, the player policy trains against a fixed enemy policy.",
    )
    ap.add_argument(
        "--snapshot-interval",
        type=int,
        default=10,
        help="Save a historical checkpoint to the opponent pool every N updates. Set to 0 to disable.",
    )
    ap.add_argument(
        "--pool-opponent-prob",
        type=float,
        default=0.5,
        help="Probability of sampling a frozen opponent from the pool instead of using current-policy self-play.",
    )
    ap.add_argument(
        "--self-play-prob",
        type=float,
        default=None,
        help="Explicit probability of using current-policy self-play. If omitted, uses 1 - pool-opponent-prob.",
    )
    ap.add_argument(
        "--max-pool-size",
        type=int,
        default=20,
        help="Maximum number of snapshots to keep per phase in the opponent pool. Set to 0 to disable size pruning.",
    )
    ap.add_argument(
        "--keep-every",
        type=int,
        default=1,
        help="When pruning, only keep snapshots whose update number is a multiple of this value.",
    )
    ap.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Run evaluation for best-model tracking every N updates. Set to 0 to disable.",
    )
    ap.add_argument(
        "--best-eval-episodes",
        type=int,
        default=6,
        help="Number of episodes to run when scoring a candidate best model.",
    )
    ap.add_argument(
        "--promotion-eval-episodes",
        type=int,
        default=8,
        help="Number of episodes to run when checking whether the current phase can advance.",
    )
    ap.add_argument(
        "--policy-arch",
        type=str,
        default="recurrent",
        choices=("feedforward", "recurrent"),
        help="Training policy architecture.",
    )
    ap.add_argument(
        "--layout",
        type=str,
        default="1v1",
        choices=("1v1", "1v2", "2v2"),
        help="Environment layout to train on.",
    )
    return ap.parse_args()


def make_algo(
    device: torch.device, obs_dim: int, act_dim: int, policy_arch: str
) -> tuple[PolicyModel, PPOConfig, PPO]:
    if policy_arch == "recurrent":
        model: PolicyModel = RecurrentActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=128, recurrent_hidden=128).to(device)
    else:
        model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=128).to(device)

    cfg = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        lr=3e-4,
        max_grad_norm=0.5,
        rollout_steps=2048,
        minibatch_size=256,
        update_epochs=8,
        sequence_length=32,
        sequences_per_batch=8,
    )

    algo = PPO(model=model, cfg=cfg, device=device)
    return model, cfg, algo


def make_rollout_buffer(
    model: PolicyModel,
    size: int,
    obs_dim: int,
    device: torch.device,
) -> RolloutBuffer | RecurrentRolloutBuffer:
    if isinstance(model, RecurrentActorCritic):
        return RecurrentRolloutBuffer(size=size, obs_dim=obs_dim, recurrent_hidden=model.recurrent_hidden, device=device)
    return RolloutBuffer(size=size, obs_dim=obs_dim, device=device)


def merge_self_play_buffers(
    train_buf: RolloutBuffer | RecurrentRolloutBuffer,
    agent_bufs: dict[str, RolloutBuffer | RecurrentRolloutBuffer],
    ordered_agent_ids: tuple[str, ...],
) -> None:
    train_buf.reset()
    offset = 0
    for agent_id in ordered_agent_ids:
        src_buf = agent_bufs[agent_id]
        size = src_buf.size
        next_offset = offset + size
        train_buf.obs[offset:next_offset] = src_buf.obs
        train_buf.act[offset:next_offset] = src_buf.act
        train_buf.rew[offset:next_offset] = src_buf.rew
        train_buf.done[offset:next_offset] = src_buf.done
        train_buf.logp[offset:next_offset] = src_buf.logp
        train_buf.val[offset:next_offset] = src_buf.val
        train_buf.adv[offset:next_offset] = src_buf.adv
        train_buf.ret[offset:next_offset] = src_buf.ret
        if isinstance(train_buf, RecurrentRolloutBuffer) and isinstance(src_buf, RecurrentRolloutBuffer):
            train_buf.episode_start[offset:next_offset] = src_buf.episode_start
            train_buf.init_h[offset:next_offset] = src_buf.init_h
            train_buf.init_c[offset:next_offset] = src_buf.init_c
        offset = next_offset
    train_buf.ptr = train_buf.size
    train_buf.full = True


def copy_buffer(dst_buf: RolloutBuffer | RecurrentRolloutBuffer, src_buf: RolloutBuffer | RecurrentRolloutBuffer) -> None:
    dst_buf.reset()
    size = src_buf.size
    dst_buf.obs[:size] = src_buf.obs
    dst_buf.act[:size] = src_buf.act
    dst_buf.rew[:size] = src_buf.rew
    dst_buf.done[:size] = src_buf.done
    dst_buf.logp[:size] = src_buf.logp
    dst_buf.val[:size] = src_buf.val
    dst_buf.adv[:size] = src_buf.adv
    dst_buf.ret[:size] = src_buf.ret
    if isinstance(dst_buf, RecurrentRolloutBuffer) and isinstance(src_buf, RecurrentRolloutBuffer):
        dst_buf.episode_start[:size] = src_buf.episode_start
        dst_buf.init_h[:size] = src_buf.init_h
        dst_buf.init_c[:size] = src_buf.init_c
    dst_buf.ptr = size
    dst_buf.full = True


class PolicyRunner:
    def __init__(self, model: PolicyModel, device: torch.device) -> None:
        self.model = model
        self.device = device

    def initial_state(self) -> HiddenState | None:
        if isinstance(self.model, RecurrentActorCritic):
            return self.model.initial_state(batch_size=1, device=self.device)
        return None

    def sample(self, obs: np.ndarray, state: HiddenState | None = None) -> tuple[int, float, float, HiddenState | None]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if isinstance(self.model, RecurrentActorCritic):
            with torch.no_grad():
                action_t, logp_t, val_t, next_state = self.model.act(obs_t, state)
            return int(action_t.item()), float(logp_t.item()), float(val_t.item()), next_state
        with torch.no_grad():
            action_t, logp_t, val_t = self.model.act(obs_t)
        return int(action_t.item()), float(logp_t.item()), float(val_t.item()), None

    def value(self, obs: np.ndarray, state: HiddenState | None = None) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if isinstance(self.model, RecurrentActorCritic):
            with torch.no_grad():
                _, val_t, _ = self.model(obs_t, state)
            return float(val_t.item())
        with torch.no_grad():
            _, val_t = self.model(obs_t)
        return float(val_t.item())


class SelfPlayCollector:
    def __init__(
        self,
        learner_runner: PolicyRunner,
        opponent_runner: PolicyRunner,
        agent_bufs: dict[str, RolloutBuffer | RecurrentRolloutBuffer],
        train_agent_ids: tuple[str, ...],
        controlled_agent_ids: tuple[str, ...],
        learner_team: str,
        team_by_agent: dict[str, str],
    ) -> None:
        self.learner_runner = learner_runner
        self.opponent_runner = opponent_runner
        self.agent_bufs = agent_bufs
        self.train_agent_ids = train_agent_ids
        self.controlled_agent_ids = controlled_agent_ids
        self.learner_team = learner_team
        self.team_by_agent = team_by_agent
        self.states: dict[str, HiddenState | None] = {}
        self.episode_start: dict[str, bool] = {}

    def runner_for(self, agent_id: str) -> PolicyRunner:
        if self.team_by_agent[agent_id] == self.learner_team:
            return self.learner_runner
        return self.opponent_runner

    def reset(self) -> None:
        for buf in self.agent_bufs.values():
            buf.reset()
        self.states = {
            agent_id: self.runner_for(agent_id).initial_state()
            for agent_id in self.controlled_agent_ids
        }
        self.episode_start = {agent_id: True for agent_id in self.controlled_agent_ids}

    def sample_actions(
        self,
        obs_by_agent: dict[str, np.ndarray],
    ) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, HiddenState | None]]:
        sampled_actions: dict[str, int] = {}
        sampled_logps: dict[str, float] = {}
        sampled_vals: dict[str, float] = {}
        next_states: dict[str, HiddenState | None] = {}
        for agent_id in self.controlled_agent_ids:
            runner = self.runner_for(agent_id)
            action, logp, val, next_state = runner.sample(obs_by_agent[agent_id], self.states[agent_id])
            sampled_actions[agent_id] = action
            sampled_logps[agent_id] = logp
            sampled_vals[agent_id] = val
            next_states[agent_id] = next_state
        return sampled_actions, sampled_logps, sampled_vals, next_states

    def store_step(
        self,
        obs_by_agent: dict[str, np.ndarray],
        sampled_actions: dict[str, int],
        sampled_logps: dict[str, float],
        sampled_vals: dict[str, float],
        next_states: dict[str, HiddenState | None],
        rewards: dict[str, float],
        done: bool,
    ) -> dict[str, float]:
        for agent_id in self.train_agent_ids:
            buf = self.agent_bufs[agent_id]
            if isinstance(buf, RecurrentRolloutBuffer):
                buf.add(
                    obs=obs_by_agent[agent_id],
                    act=sampled_actions[agent_id],
                    rew=rewards[agent_id],
                    done=done,
                    episode_start=self.episode_start[agent_id],
                    logp=sampled_logps[agent_id],
                    val=sampled_vals[agent_id],
                    state=self.states[agent_id],
                )
            else:
                buf.add(
                    obs=obs_by_agent[agent_id],
                    act=sampled_actions[agent_id],
                    rew=rewards[agent_id],
                    done=done,
                    logp=sampled_logps[agent_id],
                    val=sampled_vals[agent_id],
                )
        self.states = {agent_id: next_states[agent_id] for agent_id in self.controlled_agent_ids}
        self.episode_start = {agent_id: False for agent_id in self.controlled_agent_ids}
        if done:
            self.states = {
                agent_id: self.runner_for(agent_id).initial_state()
                for agent_id in self.controlled_agent_ids
            }
            self.episode_start = {agent_id: True for agent_id in self.controlled_agent_ids}
        return {agent_id: float(rewards[agent_id]) for agent_id in self.train_agent_ids}

    def bootstrap_values(self, obs_by_agent: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            agent_id: self.runner_for(agent_id).value(obs_by_agent[agent_id], self.states[agent_id])
            for agent_id in self.train_agent_ids
        }

    def finalize_rollouts(
        self,
        train_buf: RolloutBuffer | RecurrentRolloutBuffer,
        gamma: float,
        lam: float,
        obs_by_agent: dict[str, np.ndarray],
    ) -> None:
        last_values = self.bootstrap_values(obs_by_agent)
        for agent_id in self.train_agent_ids:
            self.agent_bufs[agent_id].compute_gae(last_val=last_values[agent_id], gamma=gamma, lam=lam)
        if len(self.train_agent_ids) == 1:
            copy_buffer(train_buf, self.agent_bufs[self.train_agent_ids[0]])
        else:
            merge_self_play_buffers(train_buf, self.agent_bufs, self.train_agent_ids)


def run_rollout(
    env: TankEnv,
    collector: SelfPlayCollector,
    cfg: PPOConfig,
    obs_by_agent: dict[str, np.ndarray],
    global_step: int,
    episode_team_returns: dict[str, float],
    episode_stats: dict[str, list[float] | list[int]],
    phase: int,
    learner_team: str,
) -> tuple[dict[str, np.ndarray], int, dict[str, float]]:
    for _ in range(cfg.rollout_steps):
        global_step += 1
        sampled_actions, sampled_logps, sampled_vals, next_states = collector.sample_actions(obs_by_agent)
        next_obs_by_agent, rewards, done, info = env.step(sampled_actions)
        reward_delta = collector.store_step(
            obs_by_agent=obs_by_agent,
            sampled_actions=sampled_actions,
            sampled_logps=sampled_logps,
            sampled_vals=sampled_vals,
            next_states=next_states,
            rewards=rewards,
            done=done,
        )
        for team_id, value in info["team_rewards"].items():
            episode_team_returns[team_id] += float(value)
        obs_by_agent = next_obs_by_agent

        if done:
            si = info["info"]
            opponent_return = sum(value for team_id, value in episode_team_returns.items() if team_id != learner_team)
            episode_stats["returns"].append(sum(episode_team_returns.values()) / max(1, len(episode_team_returns)))
            episode_stats["learner_returns"].append(episode_team_returns[learner_team])
            episode_stats["opponent_returns"].append(opponent_return)
            episode_stats["learner_wins"].append(int(si.team_wins.get(learner_team, False)))
            episode_stats["opponent_wins"].append(int(si.winning_team is not None and si.winning_team != learner_team))
            episode_stats["draws"].append(int(si.draw))
            obs_by_agent = env.reset(phase=phase)
            episode_team_returns = {team_id: 0.0 for team_id in env.team_ids()}

    return obs_by_agent, global_step, episode_team_returns


def phase_ckpt_path(models_dir: Path, phase: int, suffix: str) -> Path:
    return models_dir / f"ppo_phase{phase}_{suffix}.pt"


def opponent_pool_dir(models_dir: Path, phase: int) -> Path:
    return models_dir / "opponent_pool" / f"phase_{phase}"


def opponent_snapshot_path(models_dir: Path, phase: int, updates: int) -> Path:
    return opponent_pool_dir(models_dir, phase) / f"ppo_phase{phase}_upd{updates:06d}.pt"


def save_opponent_snapshot(
    models_dir: Path,
    model: PolicyModel,
    obs_dim: int,
    act_dim: int,
    updates: int,
    phase: int,
    agent_ids: tuple[str, ...],
    layout: str,
    cfg: PPOConfig | None = None,
) -> Path:
    pool_dir = opponent_pool_dir(models_dir, phase)
    pool_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = opponent_snapshot_path(models_dir, phase, updates)
    save_ckpt(snapshot_path, model, obs_dim, act_dim, updates, phase, agent_ids=agent_ids, layout=layout, cfg=cfg)
    return snapshot_path


def list_opponent_snapshots(models_dir: Path, phase: int) -> list[Path]:
    pool_dir = opponent_pool_dir(models_dir, phase)
    if not pool_dir.exists():
        return []
    return sorted(path for path in pool_dir.glob("*.pt") if path.is_file())


def list_compatible_opponent_snapshots(
    models_dir: Path,
    phase: int,
    obs_dim: int,
    act_dim: int,
    layout: str,
    agent_ids: tuple[str, ...],
) -> list[Path]:
    compatible: list[Path] = []
    for path in list_opponent_snapshots(models_dir, phase):
        ckpt = torch.load(path, map_location="cpu")
        if int(ckpt.get("obs_dim", -1)) != obs_dim:
            continue
        if int(ckpt.get("act_dim", -1)) != act_dim:
            continue
        if str(ckpt.get("layout", "1v1")) != layout:
            continue
        if tuple(ckpt.get("agent_ids", [])) != agent_ids:
            continue
        compatible.append(path)
    return compatible


def sample_opponent_snapshot(models_dir: Path, phase: int, rng: np.random.Generator) -> Path | None:
    snapshots = list_opponent_snapshots(models_dir, phase)
    if not snapshots:
        return None
    idx = int(rng.integers(0, len(snapshots)))
    return snapshots[idx]


def snapshot_update_number(path: Path) -> int:
    stem = path.stem
    marker = "_upd"
    if marker not in stem:
        return -1
    suffix = stem.split(marker, 1)[1]
    return int(suffix) if suffix.isdigit() else -1


def prune_opponent_pool(models_dir: Path, phase: int, max_pool_size: int, keep_every: int) -> list[Path]:
    snapshots = list_opponent_snapshots(models_dir, phase)
    if not snapshots:
        return []

    kept = [
        path
        for path in snapshots
        if keep_every <= 1 or (snapshot_update_number(path) >= 0 and snapshot_update_number(path) % keep_every == 0)
    ]
    if max_pool_size > 0 and len(kept) > max_pool_size:
        kept = kept[-max_pool_size:]

    kept_set = set(kept)
    removed: list[Path] = []
    for path in snapshots:
        if path not in kept_set:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def resolve_opponent_mix(args: argparse.Namespace) -> dict[str, Any]:
    if args.snapshot_interval < 0:
        raise ValueError("--snapshot-interval must be >= 0")
    if args.eval_interval < 0:
        raise ValueError("--eval-interval must be >= 0")
    if args.best_eval_episodes <= 0:
        raise ValueError("--best-eval-episodes must be >= 1")
    if args.promotion_eval_episodes <= 0:
        raise ValueError("--promotion-eval-episodes must be >= 1")
    if args.max_pool_size < 0:
        raise ValueError("--max-pool-size must be >= 0")
    if args.keep_every <= 0:
        raise ValueError("--keep-every must be >= 1")
    if not 0.0 <= args.pool_opponent_prob <= 1.0:
        raise ValueError("--pool-opponent-prob must be between 0 and 1")
    if args.self_play_prob is not None and not 0.0 <= args.self_play_prob <= 1.0:
        raise ValueError("--self-play-prob must be between 0 and 1")

    if args.opponent_checkpoint:
        return {
            "mode": "fixed",
            "self_play_prob": 0.0,
            "pool_opponent_prob": 0.0,
        }

    self_play_prob = 1.0 - args.pool_opponent_prob if args.self_play_prob is None else args.self_play_prob
    total_prob = self_play_prob + args.pool_opponent_prob
    if total_prob <= 0.0:
        raise ValueError("self-play and pool opponent probabilities cannot both be 0")
    if args.self_play_prob is not None and not np.isclose(total_prob, 1.0, atol=1e-6):
        raise ValueError("--self-play-prob + --pool-opponent-prob must equal 1")

    self_play_prob /= total_prob
    pool_opponent_prob = args.pool_opponent_prob / total_prob
    mode = "mixed" if pool_opponent_prob > 0.0 else "self_play"
    return {
        "mode": mode,
        "self_play_prob": self_play_prob,
        "pool_opponent_prob": pool_opponent_prob,
    }


def should_advance_phase(
    phase: int,
    single_phase: bool,
    phase_updates: int,
    min_updates_per_phase: int,
    eval_player_win_rate: float,
    advance_win_rate: float,
) -> bool:
    if single_phase or phase >= 2:
        return False
    if phase_updates < min_updates_per_phase:
        return False
    return eval_player_win_rate >= advance_win_rate


def infer_policy_arch_from_checkpoint(path: Path) -> str:
    ckpt = torch.load(path, map_location="cpu")
    return "recurrent" if ckpt.get("policy_type") == "recurrent_shared_policy" else "feedforward"


def maybe_make_pool_opponent_runner(
    opponent_mix: dict[str, Any],
    models_dir: Path,
    phase: int,
    obs_dim: int,
    act_dim: int,
    layout: str,
    agent_ids: tuple[str, ...],
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[PolicyRunner | None, str | None]:
    if opponent_mix["mode"] == "fixed":
        return None, None
    if opponent_mix["pool_opponent_prob"] <= 0.0:
        return None, None
    if rng.random() >= float(opponent_mix["pool_opponent_prob"]):
        return None, None

    snapshots = list_compatible_opponent_snapshots(models_dir, phase, obs_dim, act_dim, layout, agent_ids)
    if not snapshots:
        return None, None

    shuffled_idxs = rng.permutation(len(snapshots))
    for idx in shuffled_idxs:
        snapshot_path = snapshots[int(idx)]
        ckpt = torch.load(snapshot_path, map_location=device)
        ckpt_obs_dim = int(ckpt.get("obs_dim", -1))
        ckpt_act_dim = int(ckpt.get("act_dim", -1))
        if ckpt_obs_dim != obs_dim or ckpt_act_dim != act_dim:
            continue

        opponent_model, _ = load_policy(str(snapshot_path), device)
        label = f"{snapshot_path.name} (ckpt_v={ckpt.get('checkpoint_version', 1)})"
        return PolicyRunner(model=opponent_model, device=device), label

    return None, None


def classify_opponent_label(label: str) -> str:
    if label == "current_policy":
        return "self_play"
    if label.endswith(".pt") or ".pt (" in label:
        return "frozen"
    return "other"


def build_best_eval_opponents(
    models_dir: Path,
    phase: int,
    learner_runner: PolicyRunner,
    obs_dim: int,
    act_dim: int,
    layout: str,
    agent_ids: tuple[str, ...],
    device: torch.device,
) -> list[tuple[str, PolicyRunner]]:
    opponents: list[tuple[str, PolicyRunner]] = [("current_policy", learner_runner)]
    snapshots = list_compatible_opponent_snapshots(models_dir, phase, obs_dim, act_dim, layout, agent_ids)
    if snapshots:
        latest_snapshot = snapshots[-1]
        opponent_model, _ = load_policy(str(latest_snapshot), device)
        opponents.append((latest_snapshot.name, PolicyRunner(opponent_model, device)))
    return opponents


def evaluate_best_candidate(
    phase: int,
    episodes: int,
    learner_model: PolicyModel,
    learner_runner: PolicyRunner,
    models_dir: Path,
    obs_dim: int,
    act_dim: int,
    layout: str,
    agent_ids: tuple[str, ...],
    device: torch.device,
) -> dict[str, float]:
    if layout != "1v1":
        return {
            "score": 0.0,
            "player_win_rate": 0.0,
            "enemy_win_rate": 0.0,
            "draw_rate": 0.0,
            "avg_player_return": 0.0,
        }
    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    aggregate = {
        "score": 0.0,
        "player_win_rate": 0.0,
        "enemy_win_rate": 0.0,
        "draw_rate": 0.0,
        "avg_player_return": 0.0,
    }
    opponents = build_best_eval_opponents(
        models_dir=models_dir,
        phase=phase,
        learner_runner=learner_runner,
        obs_dim=obs_dim,
        act_dim=act_dim,
        layout=layout,
        agent_ids=agent_ids,
        device=device,
    )
    for _, opponent_runner in opponents:
        metrics = run_match_series(
            env=env,
            player_model=learner_model,
            enemy_model=opponent_runner.model,
            episodes=episodes,
            device=device,
            phase=phase,
        )
        aggregate["score"] += metrics["player_win_rate"] - metrics["enemy_win_rate"]
        aggregate["player_win_rate"] += metrics["player_win_rate"]
        aggregate["enemy_win_rate"] += metrics["enemy_win_rate"]
        aggregate["draw_rate"] += metrics["draw_rate"]
        aggregate["avg_player_return"] += metrics["avg_player_return"]

    count = float(len(opponents))
    for key in aggregate:
        aggregate[key] /= count
    return aggregate


def evaluate_phase_promotion(
    phase: int,
    episodes: int,
    learner_model: PolicyModel,
    learner_runner: PolicyRunner,
    models_dir: Path,
    obs_dim: int,
    act_dim: int,
    layout: str,
    agent_ids: tuple[str, ...],
    device: torch.device,
) -> dict[str, float]:
    return evaluate_best_candidate(
        phase=phase,
        episodes=episodes,
        learner_model=learner_model,
        learner_runner=learner_runner,
        models_dir=models_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        layout=layout,
        agent_ids=agent_ids,
        device=device,
    )


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    opponent_mix = resolve_opponent_mix(args)

    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12, layout=args.layout)
    obs_by_agent = env.reset(phase=args.phase)

    agent_ids = tuple(env.agent_ids)
    learner_team = env.team_of("player") if "player" in agent_ids else env.team_of(agent_ids[0])
    obs_dim = int(obs_by_agent[agent_ids[0]].shape[0])
    act_dim = 6
    policy_arch = args.policy_arch
    if args.resume:
        policy_arch = infer_policy_arch_from_checkpoint(Path(args.resume))

    model, cfg, algo = make_algo(
        device=device,
        obs_dim=obs_dim,
        act_dim=act_dim,
        policy_arch=policy_arch,
    )
    train_both_sides = opponent_mix["mode"] != "fixed"
    train_agent_ids = agent_ids if train_both_sides else tuple(agent_id for agent_id in agent_ids if env.team_of(agent_id) == learner_team)
    agent_bufs = {
        agent_id: make_rollout_buffer(model, cfg.rollout_steps, obs_dim, device)
        for agent_id in train_agent_ids
    }
    train_buf = make_rollout_buffer(model, cfg.rollout_steps * len(train_agent_ids), obs_dim, device)
    learner_runner = PolicyRunner(model=model, device=device)
    static_opponent_runner: PolicyRunner | None = None
    opponent_label = "current_policy"
    if args.opponent_checkpoint:
        opponent_model, opponent_ckpt = load_policy(args.opponent_checkpoint, device)
        opponent_label = Path(args.opponent_checkpoint).name
        print(
            f"Loaded frozen opponent from {args.opponent_checkpoint} "
            f"(mode={opponent_ckpt.get('training_mode', 'unknown')}, ckpt_v={opponent_ckpt.get('checkpoint_version', 1)})"
        )
        static_opponent_runner = PolicyRunner(model=opponent_model, device=device)

    models_dir = Path(__file__).resolve().parent.parent / "scripts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    current_phase = args.phase
    total_updates = 0

    if args.resume:
        total_updates, resumed_phase, ckpt = load_ckpt(Path(args.resume), model, device)
        current_phase = resumed_phase
        obs_by_agent = env.reset(phase=current_phase)
        training_mode = ckpt.get("training_mode", "unknown")
        version = ckpt.get("checkpoint_version", 1)
        print(
            f"Resumed from {args.resume} at phase={current_phase} after updates={total_updates} "
            f"(mode={training_mode}, ckpt_v={version})"
        )

    episode_stats: dict[str, list[Any]] = {
        "returns": [],
        "learner_wins": [],
        "opponent_wins": [],
        "draws": [],
        "learner_returns": [],
        "opponent_returns": [],
        "opponent_types": [],
    }

    ep_ret = {team_id: 0.0 for team_id in env.team_ids()}
    global_step = 0
    phase_updates = 0
    best_eval_score = -float("inf")
    promotion_eval_player_wr = 0.0

    print("PPO config:", asdict(cfg))
    print(
        f"Training mode: {'single phase' if args.single_phase else 'curriculum'} | "
        f"start_phase={current_phase} | advance_win_rate={args.advance_win_rate:.2f} | "
        f"min_updates_per_phase={args.min_updates_per_phase} | "
        f"opponent={'self_play' if train_both_sides else opponent_label} | "
        f"policy_arch={policy_arch} | "
        f"layout={args.layout} | "
        f"snapshot_interval={args.snapshot_interval} | "
        f"mix_mode={opponent_mix['mode']} | "
        f"self_play_prob={opponent_mix['self_play_prob']:.2f} | "
        f"pool_opponent_prob={opponent_mix['pool_opponent_prob']:.2f} | "
        f"max_pool_size={args.max_pool_size} | "
        f"keep_every={args.keep_every} | "
        f"eval_interval={args.eval_interval} | "
        f"best_eval_episodes={args.best_eval_episodes} | "
        f"promotion_eval_episodes={args.promotion_eval_episodes}"
    )

    try:
        while True:
            phase = current_phase
            round_opponent_runner = static_opponent_runner
            round_train_both_sides = train_both_sides
            round_opponent_label = opponent_label
            if static_opponent_runner is None:
                # Apply the validated opponent-mix policy once per update, not ad hoc inside the rollout.
                sampled_runner, sampled_label = maybe_make_pool_opponent_runner(
                    opponent_mix=opponent_mix,
                    models_dir=models_dir,
                    phase=phase,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    layout=args.layout,
                    agent_ids=agent_ids,
                    device=device,
                    rng=rng,
                )
                if sampled_runner is not None:
                    round_opponent_runner = sampled_runner
                    round_train_both_sides = False
                    round_opponent_label = sampled_label or "pool_opponent"
                else:
                    round_opponent_runner = learner_runner
                    round_train_both_sides = bool(opponent_mix["self_play_prob"] > 0.0)
                    round_opponent_label = "current_policy"

            # Switch between live self-play and frozen historical opponents per update.
            collector = SelfPlayCollector(
                learner_runner=learner_runner,
                opponent_runner=round_opponent_runner or learner_runner,
                agent_bufs=agent_bufs,
                train_agent_ids=agent_ids if round_train_both_sides else tuple(agent_id for agent_id in agent_ids if env.team_of(agent_id) == learner_team),
                controlled_agent_ids=agent_ids,
                learner_team=learner_team,
                team_by_agent={agent_id: env.team_of(agent_id) for agent_id in agent_ids},
            )
            collector.reset()
            # Record which opponent regime drove this update so training logs show actual league mix.
            episode_stats["opponent_types"].append(classify_opponent_label(round_opponent_label))
            obs_by_agent, global_step, ep_ret = run_rollout(
                env=env,
                collector=collector,
                cfg=cfg,
                obs_by_agent=obs_by_agent,
                global_step=global_step,
                episode_team_returns=ep_ret,
                episode_stats=episode_stats,
                phase=phase,
                learner_team=learner_team,
            )
            collector.finalize_rollouts(train_buf=train_buf, gamma=cfg.gamma, lam=cfg.lam, obs_by_agent=obs_by_agent)

            metrics = algo.update(train_buf)
            total_updates += 1
            phase_updates += 1

            mean_ret10 = float(np.mean(episode_stats["returns"][-10:])) if episode_stats["returns"] else 0.0
            wr100 = float(np.mean(episode_stats["learner_wins"][-100:])) if episode_stats["learner_wins"] else 0.0
            lr100 = float(np.mean(episode_stats["opponent_wins"][-100:])) if episode_stats["opponent_wins"] else 0.0
            dr100 = float(np.mean(episode_stats["draws"][-100:])) if episode_stats["draws"] else 0.0
            player_ret10 = float(np.mean(episode_stats["learner_returns"][-10:])) if episode_stats["learner_returns"] else 0.0
            enemy_ret10 = float(np.mean(episode_stats["opponent_returns"][-10:])) if episode_stats["opponent_returns"] else 0.0
            recent_opponents = episode_stats["opponent_types"][-20:]
            self_play_rate20 = (
                sum(1 for value in recent_opponents if value == "self_play") / len(recent_opponents)
                if recent_opponents
                else 0.0
            )
            frozen_rate20 = (
                sum(1 for value in recent_opponents if value == "frozen") / len(recent_opponents)
                if recent_opponents
                else 0.0
            )

            print(
                f"phase={phase} upd={total_updates:04d} phase_upd={phase_updates:04d} "
                f"env_steps={global_step:07d} samples={train_buf.ptr:04d} mean_ret10={mean_ret10:7.3f} "
                f"player_ret10={player_ret10:7.3f} enemy_ret10={enemy_ret10:7.3f} "
                f"player_wr100={wr100:5.2f} enemy_wr100={lr100:5.2f} dr100={dr100:5.2f} "
                f"self_play_rate20={self_play_rate20:4.2f} frozen_rate20={frozen_rate20:4.2f} "
                f"opp={round_opponent_label} "
                f"pi={metrics['pi_loss']:.3f} v={metrics['v_loss']:.3f} "
                f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.3f}"
            )

            if total_updates % 5 == 0:
                save_ckpt(
                    phase_ckpt_path(models_dir, phase, "last"),
                    model,
                    obs_dim,
                    act_dim,
                    total_updates,
                    phase,
                    agent_ids=agent_ids,
                    layout=args.layout,
                    cfg=cfg,
                )

            # Keep a time-ordered pool of older policies for future opponent sampling.
            if args.snapshot_interval > 0 and total_updates % args.snapshot_interval == 0:
                snapshot_path = save_opponent_snapshot(
                    models_dir=models_dir,
                    model=model,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    updates=total_updates,
                    phase=phase,
                    agent_ids=agent_ids,
                    layout=args.layout,
                    cfg=cfg,
                )
                print(f"Saved opponent snapshot {snapshot_path}")
                # Prune stale pool entries immediately so later sampling sees the retained set only.
                removed_paths = prune_opponent_pool(
                    models_dir=models_dir,
                    phase=phase,
                    max_pool_size=args.max_pool_size,
                    keep_every=args.keep_every,
                )
                if removed_paths:
                    print(f"Pruned {len(removed_paths)} opponent snapshots from phase={phase}")

            if args.eval_interval > 0 and total_updates % args.eval_interval == 0:
                eval_metrics = evaluate_best_candidate(
                    phase=phase,
                    episodes=args.best_eval_episodes,
                    learner_model=model,
                    learner_runner=learner_runner,
                    models_dir=models_dir,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    layout=args.layout,
                    agent_ids=agent_ids,
                    device=device,
                )
                print(
                    f"best_eval score={eval_metrics['score']:.3f} "
                    f"player_wr={eval_metrics['player_win_rate']:.2f} "
                    f"enemy_wr={eval_metrics['enemy_win_rate']:.2f} "
                    f"draw_rate={eval_metrics['draw_rate']:.2f} "
                    f"avg_player_return={eval_metrics['avg_player_return']:.3f}"
                )
                # Promote best only when head-to-head eval beats the previous best score.
                if eval_metrics["score"] > best_eval_score:
                    best_eval_score = eval_metrics["score"]
                    save_ckpt(
                        phase_ckpt_path(models_dir, phase, "best"),
                        model,
                        obs_dim,
                        act_dim,
                        total_updates,
                        phase,
                        agent_ids=agent_ids,
                        layout=args.layout,
                        cfg=cfg,
                    )
                promotion_eval_player_wr = eval_metrics["player_win_rate"]

            if should_advance_phase(
                phase=phase,
                single_phase=args.single_phase,
                phase_updates=phase_updates,
                min_updates_per_phase=args.min_updates_per_phase,
                eval_player_win_rate=promotion_eval_player_wr,
                advance_win_rate=args.advance_win_rate,
            ):
                promotion_metrics = evaluate_phase_promotion(
                    phase=phase,
                    episodes=args.promotion_eval_episodes,
                    learner_model=model,
                    learner_runner=learner_runner,
                    models_dir=models_dir,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    layout=args.layout,
                    agent_ids=agent_ids,
                    device=device,
                )
                promotion_eval_player_wr = promotion_metrics["player_win_rate"]
                print(
                    f"promotion_eval player_wr={promotion_metrics['player_win_rate']:.2f} "
                    f"enemy_wr={promotion_metrics['enemy_win_rate']:.2f} "
                    f"draw_rate={promotion_metrics['draw_rate']:.2f} "
                    f"score={promotion_metrics['score']:.3f}"
                )
                # Require a fresh evaluation pass before phase advancement, not just recent rollout statistics.
                if promotion_metrics["player_win_rate"] < args.advance_win_rate:
                    continue
                completed_phase_updates = phase_updates
                save_ckpt(
                    phase_ckpt_path(models_dir, phase, "last"),
                    model,
                    obs_dim,
                    act_dim,
                    total_updates,
                    phase,
                    agent_ids=agent_ids,
                    layout=args.layout,
                    cfg=cfg,
                )
                current_phase += 1
                phase_updates = 0
                best_eval_score = -float("inf")
                promotion_eval_player_wr = 0.0
                for values in episode_stats.values():
                    values.clear()
                ep_ret = {team_id: 0.0 for team_id in env.team_ids()}
                obs_by_agent = env.reset(phase=current_phase)
                print(
                    f"Promoted to phase={current_phase} after "
                    f"phase_updates={completed_phase_updates} total_updates={total_updates} "
                    f"promotion_player_wr={promotion_metrics['player_win_rate']:.2f}"
                )

    except KeyboardInterrupt:
        save_ckpt(
            phase_ckpt_path(models_dir, current_phase, "last"),
            model,
            obs_dim,
            act_dim,
            total_updates,
            current_phase,
            agent_ids=agent_ids,
            layout=args.layout,
            cfg=cfg,
        )
        print(f"Saved {phase_ckpt_path(models_dir, current_phase, 'last')}")


def main() -> None:
    run_training(parse_args())


__all__ = [
    "build_ckpt_payload",
    "build_best_eval_opponents",
    "CHECKPOINT_VERSION",
    "classify_opponent_label",
    "evaluate_best_candidate",
    "evaluate_phase_promotion",
    "list_opponent_snapshots",
    "load_ckpt",
    "main",
    "make_algo",
    "maybe_make_pool_opponent_runner",
    "merge_self_play_buffers",
    "opponent_pool_dir",
    "opponent_snapshot_path",
    "parse_args",
    "phase_ckpt_path",
    "prune_opponent_pool",
    "resolve_opponent_mix",
    "run_rollout",
    "run_training",
    "sample_opponent_snapshot",
    "save_ckpt",
    "save_opponent_snapshot",
    "snapshot_update_number",
    "PolicyRunner",
    "SelfPlayCollector",
    "should_advance_phase",
]
