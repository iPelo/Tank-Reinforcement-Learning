import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

from src.env.tank_env import TankEnv
from src.RL.buffer import RolloutBuffer
from src.RL.model import ActorCritic
from src.RL.ppo import PPO, PPOConfig


def save_ckpt(path: str, model: ActorCritic, obs_dim: int, act_dim: int, updates: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "updates": updates,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=2, choices=(0, 1, 2))
    args = ap.parse_args()

    device = torch.device("cpu")

    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    phase = args.phase
    obs = env.reset(phase=phase)

    obs_dim = int(obs.shape[0])  # 15
    act_dim = 6

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

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ep_returns = []
    ep_wins = []
    ep_losses = []
    ep_draws = []

    ep_ret = 0.0
    global_step = 0
    updates = 0

    best_wr100 = -1.0

    print("PPO config:", asdict(cfg))

    try:
        while True:
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
            updates += 1

            mean_ret10 = float(np.mean(ep_returns[-10:])) if len(ep_returns) >= 10 else (
                float(np.mean(ep_returns)) if ep_returns else 0.0
            )
            wr100 = float(np.mean(ep_wins[-100:])) if ep_wins else 0.0
            lr100 = float(np.mean(ep_losses[-100:])) if ep_losses else 0.0
            dr100 = float(np.mean(ep_draws[-100:])) if ep_draws else 0.0

            print(
                f"upd={updates:04d} steps={global_step:07d} mean_ret10={mean_ret10:7.3f} "
                f"wr100={wr100:5.2f} lr100={lr100:5.2f} dr100={dr100:5.2f} "
                f"pi={metrics['pi_loss']:.3f} v={metrics['v_loss']:.3f} "
                f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.3f}"
            )

            if updates % 5 == 0:
                save_ckpt(str(models_dir / f"ppo_phase{phase}_last.pt"), model, obs_dim, act_dim, updates)


            if wr100 > best_wr100:
                best_wr100 = wr100
                save_ckpt(str(models_dir / f"ppo_phase{phase}_best.pt"), model, obs_dim, act_dim, updates)

    except KeyboardInterrupt:
        save_ckpt(str(models_dir / f"ppo_phase{phase}_last.pt"), model, obs_dim, act_dim, updates)
        print(f"Saved {models_dir / f'ppo_phase{phase}_last.pt'}")


if __name__ == "__main__":
    main()
