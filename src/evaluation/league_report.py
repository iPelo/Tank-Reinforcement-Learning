import argparse
from pathlib import Path

import numpy as np
import torch

from src.evaluation.checkpoint_match import load_policy, run_match_series
from src.env.tank_env import TankEnv
from src.training.self_play import list_opponent_snapshots


def resolve_report_opponents(current_model: Path, phase: int, rng: np.random.Generator) -> list[tuple[str, Path]]:
    models_dir = current_model.parent
    opponents: list[tuple[str, Path]] = []

    best_path = models_dir / f"ppo_phase{phase}_best.pt"
    if best_path.exists() and best_path != current_model:
        opponents.append(("best", best_path))

    snapshots = list_opponent_snapshots(models_dir, phase)
    if snapshots:
        latest_snapshot = snapshots[-1]
        if latest_snapshot != current_model:
            opponents.append(("latest_pool", latest_snapshot))

        if len(snapshots) > 1:
            random_snapshot = snapshots[int(rng.integers(0, len(snapshots)))]
            if random_snapshot != current_model and random_snapshot != latest_snapshot:
                opponents.append(("random_pool", random_snapshot))

    if not opponents:
        opponents.append(("mirror", current_model))

    return opponents


def build_league_report(current_model_path: str, episodes: int, phase: int, seed: int) -> list[dict[str, object]]:
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    current_path = Path(current_model_path)

    player_model, player_ckpt = load_policy(str(current_path), device)
    opponents = resolve_report_opponents(current_path, phase, rng)

    report_rows: list[dict[str, object]] = []
    for label, opponent_path in opponents:
        enemy_model, enemy_ckpt = load_policy(str(opponent_path), device)
        env = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
        metrics = run_match_series(
            env=env,
            player_model=player_model,
            enemy_model=enemy_model,
            episodes=episodes,
            device=device,
            phase=phase,
        )
        report_rows.append(
            {
                "label": label,
                "player": current_path.name,
                "player_updates": player_ckpt.get("updates", 0),
                "enemy": opponent_path.name,
                "enemy_updates": enemy_ckpt.get("updates", 0),
                "metrics": metrics,
            }
        )

    return report_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-model", type=str, required=True, help="Checkpoint path for the current candidate model.")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--phase", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_league_report(
        current_model_path=args.current_model,
        episodes=args.episodes,
        phase=args.phase,
        seed=args.seed,
    )

    print(f"league_report current={Path(args.current_model).name} phase={args.phase} episodes={args.episodes}")
    # Keep the report compact so it stays readable during regular training workflows.
    for row in rows:
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"{row['label']}: "
            f"player={row['player']}@{row['player_updates']} "
            f"enemy={row['enemy']}@{row['enemy_updates']} "
            f"player_wr={metrics['player_win_rate']:.2f} "
            f"enemy_wr={metrics['enemy_win_rate']:.2f} "
            f"draw_rate={metrics['draw_rate']:.2f} "
            f"avg_player_return={metrics['avg_player_return']:.3f} "
            f"avg_steps={metrics['avg_steps']:.1f}"
        )


if __name__ == "__main__":
    main()
