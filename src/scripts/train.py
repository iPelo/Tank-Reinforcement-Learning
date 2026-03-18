import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.env.tank_env import TankEnv
from src.RL.buffer import RolloutBuffer
from src.RL.model import ActorCritic
from src.RL.ppo import PPO, PPOConfig


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


def make_algo(device: torch.device, obs_dim: int, act_dim: int) -> tuple[ActorCritic, PPOConfig, PPO, RolloutBuffer]:
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
    buf = RolloutBuffer(size=cfg.rollout_steps, obs_dim=obs_dim, device=device)
    return model, cfg, algo, buf


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


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    obs = env.reset(phase=args.phase)

    obs_dim = int(obs.shape[0])
    act_dim = 6

    model, cfg, algo, buf = make_algo(device=device, obs_dim=obs_dim, act_dim=act_dim)

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    current_phase = args.phase
    total_updates = 0

    if args.resume:
        total_updates, resumed_phase = load_ckpt(Path(args.resume), model, device)
        current_phase = resumed_phase
        obs = env.reset(phase=current_phase)
        print(f"Resumed from {args.resume} at phase={current_phase} after updates={total_updates}")

    ep_returns: list[float] = []
    ep_wins: list[int] = []
    ep_losses: list[int] = []
    ep_draws: list[int] = []

    ep_ret = 0.0
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
            buf.reset()

            for _ in range(cfg.rollout_steps):
                global_step += 1

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, logp_t, val_t = model.act(obs_t)

                action = int(action_t.item())
                logp = float(logp_t.item())
                val = float(val_t.item())

                next_obs, rew, done, info = env.step(action)
                buf.add(obs=obs, act=action, rew=rew, done=done, logp=logp, val=val)

                ep_ret += float(rew)
                obs = next_obs

                if done:
                    si = info["info"]
                    ep_returns.append(ep_ret)
                    ep_wins.append(int(si.player_win))
                    ep_losses.append(int(si.enemy_win))
                    ep_draws.append(int(si.draw))

                    obs = env.reset(phase=phase)
                    ep_ret = 0.0

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, last_val_t = model(obs_t)
            last_val = float(last_val_t.item())

            buf.compute_gae(last_val=last_val, gamma=cfg.gamma, lam=cfg.lam)

            metrics = algo.update(buf)
            total_updates += 1
            phase_updates += 1

            mean_ret10 = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
            wr100 = float(np.mean(ep_wins[-100:])) if ep_wins else 0.0
            lr100 = float(np.mean(ep_losses[-100:])) if ep_losses else 0.0
            dr100 = float(np.mean(ep_draws[-100:])) if ep_draws else 0.0

            print(
                f"phase={phase} upd={total_updates:04d} phase_upd={phase_updates:04d} "
                f"steps={global_step:07d} mean_ret10={mean_ret10:7.3f} "
                f"wr100={wr100:5.2f} lr100={lr100:5.2f} dr100={dr100:5.2f} "
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
                ep_wins.clear()
                ep_losses.clear()
                ep_draws.clear()
                ep_ret = 0.0
                obs = env.reset(phase=current_phase)
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


if __name__ == "__main__":
    main()
