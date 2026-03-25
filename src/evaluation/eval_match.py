import argparse
import time

import numpy as np
import torch

from src.env.render import PygameRenderer
from src.env.tank_env import TankEnv
from src.evaluation.checkpoint_match import act_with_policy, init_policy_runtime, load_policy


def run_random(env: TankEnv, renderer: PygameRenderer | None, episodes: int, seed: int, phase: int) -> None:
    rng = np.random.default_rng(seed)
    blue_wins: list[int] = []
    red_wins: list[int] = []
    draws: list[int] = []

    for ep in range(episodes):
        _ = env.reset(phase=phase)
        returns = {team_id: 0.0 for team_id in env.team_ids()}
        done = False

        while not done:
            actions = {agent_id: int(rng.integers(0, 6)) for agent_id in env.agent_ids}
            _, _, done, info = env.step(actions)
            for team_id, reward in info["team_rewards"].items():
                returns[team_id] += float(reward)

            if renderer is not None:
                si = info["info"]
                renderer.render(
                    env,
                    text=(
                        f"RAND {env.layout_name} ep={ep} steps={si.steps} "
                        f"Rb={returns.get('blue', 0.0):.2f} Rr={returns.get('red', 0.0):.2f}"
                    ),
                )

        si = info["info"]
        blue_wins.append(int(si.team_wins.get("blue", False)))
        red_wins.append(int(si.team_wins.get("red", False)))
        draws.append(int(si.draw))
        if len(blue_wins) > 100:
            blue_wins.pop(0)
            red_wins.pop(0)
            draws.pop(0)
        wr = sum(blue_wins) / len(blue_wins)
        lr = sum(red_wins) / len(red_wins)
        dr = sum(draws) / len(draws)

        print(
            f"RAND {env.layout_name} EP {ep:03d} | steps={si.steps:3d} "
            f"| Rb={returns.get('blue', 0.0):6.2f} Rr={returns.get('red', 0.0):6.2f} "
            f"| blue_win={si.team_wins.get('blue', False)} red_win={si.team_wins.get('red', False)} draw={si.draw}"
            f"| last100_bwr={wr:.2f} rwr={lr:.2f} dr={dr:.2f}"
        )


def run_model(env: TankEnv, renderer: PygameRenderer | None, episodes: int, model_path: str, phase: int) -> None:
    device = torch.device("cpu")
    model, ckpt = load_policy(model_path, device)
    training_mode = ckpt.get("training_mode", "unknown")
    checkpoint_version = ckpt.get("checkpoint_version", 1)
    policy_type = ckpt.get("policy_type", "unknown")

    print(
        f"Loaded checkpoint: mode={training_mode} policy={policy_type} "
        f"phase={ckpt.get('phase', phase)} updates={ckpt.get('updates', 0)} "
        f"ckpt_v={checkpoint_version}"
    )

    blue_wins: list[int] = []
    red_wins: list[int] = []
    draws: list[int] = []

    for ep in range(episodes):
        obs_by_agent = env.reset(phase=phase)
        runtimes = {agent_id: init_policy_runtime(model, device) for agent_id in env.agent_ids}
        returns = {team_id: 0.0 for team_id in env.team_ids()}
        done = False

        while not done:
            actions: dict[str, int] = {}
            for agent_id in env.agent_ids:
                obs_t = torch.as_tensor(obs_by_agent[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                actions[agent_id] = act_with_policy(model, obs_t, runtimes[agent_id])

            obs_by_agent, _, done, info = env.step(actions)
            for team_id, reward in info["team_rewards"].items():
                returns[team_id] += float(reward)

            if renderer is not None:
                si = info["info"]
                renderer.render(
                    env,
                    text=(
                        f"PPO {env.layout_name} ep={ep} steps={si.steps} "
                        f"Rb={returns.get('blue', 0.0):.2f} Rr={returns.get('red', 0.0):.2f}"
                    ),
                )

        si = info["info"]
        blue_wins.append(int(si.team_wins.get("blue", False)))
        red_wins.append(int(si.team_wins.get("red", False)))
        draws.append(int(si.draw))
        if len(blue_wins) > 100:
            blue_wins.pop(0)
            red_wins.pop(0)
            draws.pop(0)
        wr = sum(blue_wins) / len(blue_wins)
        lr = sum(red_wins) / len(red_wins)
        dr = sum(draws) / len(draws)

        print(
            f"PPO {env.layout_name} EP {ep:03d} | steps={si.steps:3d} "
            f"| Rb={returns.get('blue', 0.0):6.2f} Rr={returns.get('red', 0.0):6.2f} "
            f"| blue_win={si.team_wins.get('blue', False)} red_win={si.team_wins.get('red', False)} draw={si.draw}"
            f"| last100_bwr={wr:.2f} rwr={lr:.2f} dr={dr:.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--phase", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--layout", type=str, default="1v1", choices=("1v1", "1v2", "2v2"))
    args = ap.parse_args()

    env_cfg = TankEnv.default_config(args.layout)
    env = TankEnv(
        w=env_cfg.w,
        h=env_cfg.h,
        max_steps=env_cfg.max_steps,
        seed=0,
        wall_density=env_cfg.wall_density,
        layout=args.layout,
    )
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
