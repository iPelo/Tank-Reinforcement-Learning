import argparse
from pathlib import Path

import torch

from src.agents.policy import ActorCritic
from src.env.tank_env import TankEnv


def load_policy(model_path: str, device: torch.device) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(model_path, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=128).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run_match_series(
    env: TankEnv,
    player_model: ActorCritic,
    enemy_model: ActorCritic,
    episodes: int,
    device: torch.device,
    phase: int,
) -> dict[str, float]:
    player_wins = 0
    enemy_wins = 0
    draws = 0
    player_return = 0.0
    enemy_return = 0.0
    total_steps = 0

    for _ in range(episodes):
        obs_by_agent = env.reset(phase=phase)
        done = False

        while not done:
            actions: dict[str, int] = {}
            for agent_id, model in (("player", player_model), ("enemy", enemy_model)):
                obs_t = torch.as_tensor(obs_by_agent[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _ = model.act(obs_t)
                actions[agent_id] = int(action_t.item())

            obs_by_agent, rewards, done, info = env.step(actions)
            player_return += float(rewards["player"])
            enemy_return += float(rewards["enemy"])

        si = info["info"]
        player_wins += int(si.player_win)
        enemy_wins += int(si.enemy_win)
        draws += int(si.draw)
        total_steps += int(si.steps)

    return {
        "player_win_rate": player_wins / max(1, episodes),
        "enemy_win_rate": enemy_wins / max(1, episodes),
        "draw_rate": draws / max(1, episodes),
        "avg_player_return": player_return / max(1, episodes),
        "avg_enemy_return": enemy_return / max(1, episodes),
        "avg_steps": total_steps / max(1, episodes),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-model", type=str, required=True, help="Checkpoint path for the player-side policy.")
    ap.add_argument("--enemy-model", type=str, required=True, help="Checkpoint path for the enemy-side policy.")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--phase", type=int, default=2, choices=(0, 1, 2))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    player_model, player_ckpt = load_policy(args.player_model, device)
    enemy_model, enemy_ckpt = load_policy(args.enemy_model, device)

    env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    metrics = run_match_series(
        env=env,
        player_model=player_model,
        enemy_model=enemy_model,
        episodes=args.episodes,
        device=device,
        phase=args.phase,
    )

    print(
        f"player={Path(args.player_model).name} "
        f"(phase={player_ckpt.get('phase', args.phase)} updates={player_ckpt.get('updates', 0)})"
    )
    print(
        f"enemy={Path(args.enemy_model).name} "
        f"(phase={enemy_ckpt.get('phase', args.phase)} updates={enemy_ckpt.get('updates', 0)})"
    )
    print(
        "match_summary "
        f"episodes={args.episodes} "
        f"player_wr={metrics['player_win_rate']:.2f} "
        f"enemy_wr={metrics['enemy_win_rate']:.2f} "
        f"draw_rate={metrics['draw_rate']:.2f} "
        f"avg_player_return={metrics['avg_player_return']:.3f} "
        f"avg_enemy_return={metrics['avg_enemy_return']:.3f} "
        f"avg_steps={metrics['avg_steps']:.1f}"
    )


if __name__ == "__main__":
    main()
