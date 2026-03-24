import argparse
from pathlib import Path
from typing import Any

import torch

from src.agents.policy import ActorCritic, RecurrentActorCritic
from src.env.tank_env import TankEnv


PolicyModel = ActorCritic | RecurrentActorCritic


def load_policy(model_path: str, device: torch.device) -> tuple[PolicyModel, dict]:
    ckpt = torch.load(model_path, map_location=device)
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    policy_type = ckpt.get("policy_type", "shared_policy")
    if policy_type == "recurrent_shared_policy":
        hidden = int(ckpt.get("hidden", 128))
        recurrent_hidden = int(ckpt.get("recurrent_hidden", 128))
        model = RecurrentActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            recurrent_hidden=recurrent_hidden,
        ).to(device)
    else:
        hidden = int(ckpt.get("hidden", 128))
        model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def init_policy_runtime(model: PolicyModel, device: torch.device) -> dict[str, Any]:
    if isinstance(model, RecurrentActorCritic):
        return {"state": model.initial_state(batch_size=1, device=device)}
    return {"state": None}


def act_with_policy(model: PolicyModel, obs: torch.Tensor, runtime: dict[str, Any]) -> int:
    if isinstance(model, RecurrentActorCritic):
        action_t, _, _, next_state = model.act(obs, runtime["state"])
        runtime["state"] = next_state
        return int(action_t.item())

    action_t, _, _ = model.act(obs)
    return int(action_t.item())


def run_match_series(
    env: TankEnv,
    player_model: PolicyModel,
    enemy_model: PolicyModel,
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
        runtimes = {
            "player": init_policy_runtime(player_model, device),
            "enemy": init_policy_runtime(enemy_model, device),
        }
        done = False

        while not done:
            actions: dict[str, int] = {}
            for agent_id, model in (("player", player_model), ("enemy", enemy_model)):
                obs_t = torch.as_tensor(obs_by_agent[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                actions[agent_id] = act_with_policy(model, obs_t, runtimes[agent_id])

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
