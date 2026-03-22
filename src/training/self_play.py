import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.agents.policy import ActorCritic
from src.env.tank_env import TankEnv
from src.training.buffer import RolloutBuffer
from src.training.ppo import PPO, PPOConfig


def save_ckpt(path: Path, model: ActorCritic, obs_dim: int, act_dim: int, updates: int, phase: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "updates": updates,
            "phase": phase,
        },
        path,
    )


def load_ckpt(path: Path, model: ActorCritic, device: torch.device) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return int(ckpt.get("updates", 0)), int(ckpt.get("phase", 0))


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
    return ap.parse_args()


def make_algo(
    device: torch.device, obs_dim: int, act_dim: int
) -> tuple[ActorCritic, PPOConfig, PPO, RolloutBuffer, RolloutBuffer, RolloutBuffer]:
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
    )

    algo = PPO(model=model, cfg=cfg, device=device)
    player_buf = RolloutBuffer(size=cfg.rollout_steps, obs_dim=obs_dim, device=device)
    enemy_buf = RolloutBuffer(size=cfg.rollout_steps, obs_dim=obs_dim, device=device)
    train_buf = RolloutBuffer(size=cfg.rollout_steps * 2, obs_dim=obs_dim, device=device)
    return model, cfg, algo, player_buf, enemy_buf, train_buf


def merge_self_play_buffers(train_buf: RolloutBuffer, player_buf: RolloutBuffer, enemy_buf: RolloutBuffer) -> None:
    train_buf.reset()
    split = player_buf.size

    train_buf.obs[:split] = player_buf.obs
    train_buf.obs[split:] = enemy_buf.obs
    train_buf.act[:split] = player_buf.act
    train_buf.act[split:] = enemy_buf.act
    train_buf.rew[:split] = player_buf.rew
    train_buf.rew[split:] = enemy_buf.rew
    train_buf.done[:split] = player_buf.done
    train_buf.done[split:] = enemy_buf.done
    train_buf.logp[:split] = player_buf.logp
    train_buf.logp[split:] = enemy_buf.logp
    train_buf.val[:split] = player_buf.val
    train_buf.val[split:] = enemy_buf.val
    train_buf.adv[:split] = player_buf.adv
    train_buf.adv[split:] = enemy_buf.adv
    train_buf.ret[:split] = player_buf.ret
    train_buf.ret[split:] = enemy_buf.ret
    train_buf.ptr = train_buf.size
    train_buf.full = True


def phase_ckpt_path(models_dir: Path, phase: int, suffix: str) -> Path:
    return models_dir / f"ppo_phase{phase}_{suffix}.pt"


def should_advance_phase(
    phase: int,
    single_phase: bool,
    phase_updates: int,
    min_updates_per_phase: int,
    win_rate: float,
    advance_win_rate: float,
) -> bool:
    if single_phase or phase >= 2:
        return False
    if phase_updates < min_updates_per_phase:
        return False
    return win_rate >= advance_win_rate


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cpu")

    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    obs_by_agent = env.reset(phase=args.phase)

    obs_dim = int(obs_by_agent["player"].shape[0])
    act_dim = 6

    model, cfg, algo, player_buf, enemy_buf, train_buf = make_algo(device=device, obs_dim=obs_dim, act_dim=act_dim)

    models_dir = Path(__file__).resolve().parent.parent / "scripts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    current_phase = args.phase
    total_updates = 0

    if args.resume:
        total_updates, resumed_phase = load_ckpt(Path(args.resume), model, device)
        current_phase = resumed_phase
        obs_by_agent = env.reset(phase=current_phase)
        print(f"Resumed from {args.resume} at phase={current_phase} after updates={total_updates}")

    ep_returns: list[float] = []
    ep_player_wins: list[int] = []
    ep_enemy_wins: list[int] = []
    ep_draws: list[int] = []

    ep_ret = {"player": 0.0, "enemy": 0.0}
    global_step = 0
    phase_updates = 0
    best_wr100 = -1.0

    print("PPO config:", asdict(cfg))
    print(
        f"Training mode: {'single phase' if args.single_phase else 'curriculum'} | "
        f"start_phase={current_phase} | advance_win_rate={args.advance_win_rate:.2f} | "
        f"min_updates_per_phase={args.min_updates_per_phase}"
    )

    try:
        while True:
            phase = current_phase
            player_buf.reset()
            enemy_buf.reset()

            for _ in range(cfg.rollout_steps):
                global_step += 1
                sampled_actions: dict[str, int] = {}
                sampled_logps: dict[str, float] = {}
                sampled_vals: dict[str, float] = {}

                for agent_id in ("player", "enemy"):
                    obs_t = torch.as_tensor(obs_by_agent[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action_t, logp_t, val_t = model.act(obs_t)

                    sampled_actions[agent_id] = int(action_t.item())
                    sampled_logps[agent_id] = float(logp_t.item())
                    sampled_vals[agent_id] = float(val_t.item())

                next_obs_by_agent, rewards, done, info = env.step(sampled_actions)

                player_buf.add(
                    obs=obs_by_agent["player"],
                    act=sampled_actions["player"],
                    rew=rewards["player"],
                    done=done,
                    logp=sampled_logps["player"],
                    val=sampled_vals["player"],
                )
                enemy_buf.add(
                    obs=obs_by_agent["enemy"],
                    act=sampled_actions["enemy"],
                    rew=rewards["enemy"],
                    done=done,
                    logp=sampled_logps["enemy"],
                    val=sampled_vals["enemy"],
                )
                ep_ret["player"] += float(rewards["player"])
                ep_ret["enemy"] += float(rewards["enemy"])

                obs_by_agent = next_obs_by_agent

                if done:
                    si = info["info"]
                    ep_returns.append(0.5 * (ep_ret["player"] + ep_ret["enemy"]))
                    ep_player_wins.append(int(si.player_win))
                    ep_enemy_wins.append(int(si.enemy_win))
                    ep_draws.append(int(si.draw))

                    obs_by_agent = env.reset(phase=phase)
                    ep_ret = {"player": 0.0, "enemy": 0.0}

            last_values: dict[str, float] = {}
            for agent_id in ("player", "enemy"):
                obs_t = torch.as_tensor(obs_by_agent[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, last_val_t = model(obs_t)
                last_values[agent_id] = float(last_val_t.item())

            player_buf.compute_gae(last_val=last_values["player"], gamma=cfg.gamma, lam=cfg.lam)
            enemy_buf.compute_gae(last_val=last_values["enemy"], gamma=cfg.gamma, lam=cfg.lam)
            merge_self_play_buffers(train_buf, player_buf, enemy_buf)

            metrics = algo.update(train_buf)
            total_updates += 1
            phase_updates += 1

            mean_ret10 = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
            wr100 = float(np.mean(ep_player_wins[-100:])) if ep_player_wins else 0.0
            lr100 = float(np.mean(ep_enemy_wins[-100:])) if ep_enemy_wins else 0.0
            dr100 = float(np.mean(ep_draws[-100:])) if ep_draws else 0.0

            print(
                f"phase={phase} upd={total_updates:04d} phase_upd={phase_updates:04d} "
                f"env_steps={global_step:07d} samples={train_buf.ptr:04d} mean_ret10={mean_ret10:7.3f} "
                f"player_wr100={wr100:5.2f} enemy_wr100={lr100:5.2f} dr100={dr100:5.2f} "
                f"pi={metrics['pi_loss']:.3f} v={metrics['v_loss']:.3f} "
                f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.3f}"
            )

            if total_updates % 5 == 0:
                save_ckpt(phase_ckpt_path(models_dir, phase, "last"), model, obs_dim, act_dim, total_updates, phase)

            if wr100 > best_wr100:
                best_wr100 = wr100
                save_ckpt(phase_ckpt_path(models_dir, phase, "best"), model, obs_dim, act_dim, total_updates, phase)

            if should_advance_phase(
                phase=phase,
                single_phase=args.single_phase,
                phase_updates=phase_updates,
                min_updates_per_phase=args.min_updates_per_phase,
                win_rate=wr100,
                advance_win_rate=args.advance_win_rate,
            ):
                completed_phase_updates = phase_updates
                save_ckpt(phase_ckpt_path(models_dir, phase, "last"), model, obs_dim, act_dim, total_updates, phase)
                current_phase += 1
                phase_updates = 0
                best_wr100 = -1.0
                ep_returns.clear()
                ep_player_wins.clear()
                ep_enemy_wins.clear()
                ep_draws.clear()
                ep_ret = {"player": 0.0, "enemy": 0.0}
                obs_by_agent = env.reset(phase=current_phase)
                print(
                    f"Promoted to phase={current_phase} after "
                    f"phase_updates={completed_phase_updates} total_updates={total_updates} wr100={wr100:.2f}"
                )

    except KeyboardInterrupt:
        save_ckpt(
            phase_ckpt_path(models_dir, current_phase, "last"),
            model,
            obs_dim,
            act_dim,
            total_updates,
            current_phase,
        )
        print(f"Saved {phase_ckpt_path(models_dir, current_phase, 'last')}")


def main() -> None:
    run_training(parse_args())


__all__ = [
    "load_ckpt",
    "main",
    "make_algo",
    "merge_self_play_buffers",
    "parse_args",
    "phase_ckpt_path",
    "run_training",
    "save_ckpt",
    "should_advance_phase",
]
