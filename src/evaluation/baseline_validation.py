import argparse
from pathlib import Path

import torch

from src.evaluation.checkpoint_match import load_policy, run_match_series
from src.env.tank_env import TankEnv


def validate_against_baseline(candidate_model_path: str, baseline_model_path: str, episodes: int, phase: int) -> dict[str, object]:
    device = torch.device("cpu")
    candidate_model, candidate_ckpt = load_policy(candidate_model_path, device)
    baseline_model, baseline_ckpt = load_policy(baseline_model_path, device)

    env_a = TankEnv(w=15, h=15, max_steps=200, seed=0, wall_density=0.12)
    candidate_as_player = run_match_series(
        env=env_a,
        player_model=candidate_model,
        enemy_model=baseline_model,
        episodes=episodes,
        device=device,
        phase=phase,
    )

    env_b = TankEnv(w=15, h=15, max_steps=200, seed=1, wall_density=0.12)
    baseline_as_player = run_match_series(
        env=env_b,
        player_model=baseline_model,
        enemy_model=candidate_model,
        episodes=episodes,
        device=device,
        phase=phase,
    )

    # Evaluate both seat assignments so the comparison is less sensitive to role placement bias.
    candidate_overall_wr = 0.5 * (
        candidate_as_player["player_win_rate"] + baseline_as_player["enemy_win_rate"]
    )
    baseline_overall_wr = 0.5 * (
        candidate_as_player["enemy_win_rate"] + baseline_as_player["player_win_rate"]
    )
    draw_rate = 0.5 * (candidate_as_player["draw_rate"] + baseline_as_player["draw_rate"])
    candidate_return = 0.5 * (
        candidate_as_player["avg_player_return"] + baseline_as_player["avg_enemy_return"]
    )
    baseline_return = 0.5 * (
        candidate_as_player["avg_enemy_return"] + baseline_as_player["avg_player_return"]
    )

    return {
        "candidate": {
            "name": Path(candidate_model_path).name,
            "updates": candidate_ckpt.get("updates", 0),
            "policy_type": candidate_ckpt.get("policy_type", "unknown"),
        },
        "baseline": {
            "name": Path(baseline_model_path).name,
            "updates": baseline_ckpt.get("updates", 0),
            "policy_type": baseline_ckpt.get("policy_type", "unknown"),
        },
        "candidate_as_player": candidate_as_player,
        "baseline_as_player": baseline_as_player,
        "summary": {
            "candidate_overall_win_rate": candidate_overall_wr,
            "baseline_overall_win_rate": baseline_overall_wr,
            "draw_rate": draw_rate,
            "candidate_avg_return": candidate_return,
            "baseline_avg_return": baseline_return,
            "candidate_advantage": candidate_overall_wr - baseline_overall_wr,
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-model", type=str, required=True, help="Model being validated.")
    ap.add_argument("--baseline-model", type=str, required=True, help="Reference baseline model.")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--phase", type=int, default=2, choices=(0, 1, 2))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_against_baseline(
        candidate_model_path=args.candidate_model,
        baseline_model_path=args.baseline_model,
        episodes=args.episodes,
        phase=args.phase,
    )
    summary = report["summary"]
    candidate = report["candidate"]
    baseline = report["baseline"]
    print(
        f"baseline_validation candidate={candidate['name']}@{candidate['updates']}[{candidate['policy_type']}] "
        f"baseline={baseline['name']}@{baseline['updates']}[{baseline['policy_type']}] "
        f"episodes={args.episodes} phase={args.phase}"
    )
    print(
        f"summary candidate_wr={summary['candidate_overall_win_rate']:.2f} "
        f"baseline_wr={summary['baseline_overall_win_rate']:.2f} "
        f"draw_rate={summary['draw_rate']:.2f} "
        f"candidate_return={summary['candidate_avg_return']:.3f} "
        f"baseline_return={summary['baseline_avg_return']:.3f} "
        f"advantage={summary['candidate_advantage']:.3f}"
    )
    print(
        f"candidate_as_player player_wr={report['candidate_as_player']['player_win_rate']:.2f} "
        f"enemy_wr={report['candidate_as_player']['enemy_win_rate']:.2f} "
        f"draw_rate={report['candidate_as_player']['draw_rate']:.2f}"
    )
    print(
        f"baseline_as_player player_wr={report['baseline_as_player']['player_win_rate']:.2f} "
        f"enemy_wr={report['baseline_as_player']['enemy_win_rate']:.2f} "
        f"draw_rate={report['baseline_as_player']['draw_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
