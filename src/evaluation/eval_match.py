import argparse
import time

import numpy as np
import torch

from src.env.tank_env import TankEnv
from src.env.render import PygameRenderer
from src.agents.policy import ActorCritic


def run_random(env: TankEnv, renderer: PygameRenderer | None, episodes: int, seed: int, phase: int) -> None:
    rng = np.random.default_rng(seed)
    player_wins: list[int] = []
    enemy_wins: list[int] = []
    draws: list[int] = []

    for ep in range(episodes):
        _ = env.reset(phase=phase)
        returns = {"player": 0.0, "enemy": 0.0}
        done = False

        while not done:
            actions = {
                "player": int(rng.integers(0, 6)),
                "enemy": int(rng.integers(0, 6)),
            }
            _, rewards, done, info = env.step(actions)
            returns["player"] += float(rewards["player"])
            returns["enemy"] += float(rewards["enemy"])

            if renderer is not None:
                si = info["info"]
                renderer.render(
                    env,
                    text=(
                        f"RAND ep={ep} steps={si.steps} "
                        f"Rp={returns['player']:.2f} Re={returns['enemy']:.2f}"
                    ),
                )

        si = info["info"]
        player_wins.append(int(si.player_win))
        enemy_wins.append(int(si.enemy_win))
        draws.append(int(si.draw))
        if len(player_wins) > 100:
            player_wins.pop(0)
            enemy_wins.pop(0)
            draws.pop(0)
        wr = sum(player_wins) / len(player_wins)
        lr = sum(enemy_wins) / len(enemy_wins)
        dr = sum(draws) / len(draws)

        print(
            f"RAND EP {ep:03d} | steps={si.steps:3d} "
            f"| Rp={returns['player']:6.2f} Re={returns['enemy']:6.2f} "
            f"| player_win={si.player_win} enemy_win={si.enemy_win} draw={si.draw}"
            f"| last100_pwr={wr:.2f} ewr={lr:.2f} dr={dr:.2f}"
        )


def run_model(env: TankEnv, renderer: PygameRenderer | None, episodes: int, model_path: str, phase: int) -> None:
    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device)

    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=128).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    player_wins: list[int] = []
    enemy_wins: list[int] = []
    draws: list[int] = []

    for ep in range(episodes):
        obs_by_agent = env.reset(phase=phase)
        returns = {"player": 0.0, "enemy": 0.0}
        done = False

        while not done:
            actions: dict[str, int] = {}
            for agent_id in ("player", "enemy"):
                obs_t = torch.as_tensor(obs_by_agent[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _ = model.act(obs_t)
                actions[agent_id] = int(action_t.item())

            obs_by_agent, rewards, done, info = env.step(actions)
            returns["player"] += float(rewards["player"])
            returns["enemy"] += float(rewards["enemy"])

            if renderer is not None:
                si = info["info"]
                renderer.render(
                    env,
                    text=(
                        f"PPO ep={ep} steps={si.steps} "
                        f"Rp={returns['player']:.2f} Re={returns['enemy']:.2f}"
                    ),
                )

        si = info["info"]
        player_wins.append(int(si.player_win))
        enemy_wins.append(int(si.enemy_win))
        draws.append(int(si.draw))
        if len(player_wins) > 100:
            player_wins.pop(0)
            enemy_wins.pop(0)
            draws.pop(0)
        wr = sum(player_wins) / len(player_wins)
        lr = sum(enemy_wins) / len(enemy_wins)
        dr = sum(draws) / len(draws)

        print(
            f"PPO  EP {ep:03d} | steps={si.steps:3d} "
            f"| Rp={returns['player']:6.2f} Re={returns['enemy']:6.2f} "
            f"| player_win={si.player_win} enemy_win={si.enemy_win} draw={si.draw}"
            f"| last100_pwr={wr:.2f} ewr={lr:.2f} dr={dr:.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--phase", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--model", type=str, default="")
    args = ap.parse_args()

    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    renderer = PygameRenderer(cell_size=32, fps=20) if args.render else None

    try:
        if args.model:
            run_model(env, renderer, args.episodes, args.model, args.phase)
        else:
            run_random(env, renderer, args.episodes, seed=0, phase=args.phase)
    finally:
        if renderer is not None:
            time.sleep(0.25)
            renderer.close()


if __name__ == "__main__":
    main()
