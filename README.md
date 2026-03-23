<h1 align="center">
  <a href="https://github.com/iPelo/Tank-Reinforcement-Learning">
    TANK-REINFORCEMENT-LEARNING
  </a>
</h1>

<p align="center">Train a hidden-information PPO tank AI through self-play, opponent pools, and evaluation-driven progression</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/github/stars/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Stars">
  <img src="https://img.shields.io/github/forks/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Forks">
  <img src="https://img.shields.io/github/issues/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Issues">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-PPO-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge" alt="PyTorch PPO">
  <img src="https://img.shields.io/badge/Environment-Hidden%20Info-0E7490?style=for-the-badge" alt="Hidden Information">
  <img src="https://img.shields.io/badge/Training-Self%20Play-15803D?style=for-the-badge" alt="Self Play">
  <img src="https://img.shields.io/badge/Eval-League%20Report-7C3AED?style=for-the-badge" alt="League Report">
</p>

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=iPelo&repo=Tank-Reinforcement-Learning&theme=transparent&hide_border=true" alt="Repository Widget">
</p>

---

## Table of Contents
- [Overview](#overview)
- [Current Features](#current-features)
- [Environment](#environment)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints and Opponent Pool](#checkpoints-and-opponent-pool)
- [Roadmap Status](#roadmap-status)

---

## Overview

**Tank-Reinforcement-Learning** is a reinforcement learning project where a PPO policy learns tank combat through **self-play** in a **15x15 grid world** with random walls.

The current version is no longer a simple agent-vs-scripted-bot prototype. It now includes:

- Symmetric `AI vs AI` tank combat
- Hidden-information observations instead of always-visible enemy state
- Shared-policy self-play and frozen-opponent training
- Opponent pool snapshot saving and sampling
- Checkpoint-vs-checkpoint evaluation
- League-style reporting and evaluation-driven best model tracking

---

## Current Features

- **Symmetric environment**: both tanks follow the same rules, actions, rewards, and observation shape
- **Hidden-information setup**: tanks only receive direct enemy directional features when line of sight exists
- **Self-play trainer**: current policy can train against itself, a fixed checkpoint, or sampled pool opponents
- **Opponent pool**: historical snapshots are saved under phase-specific directories for later reuse
- **Best-model tracking**: `best` checkpoints are updated from evaluation results, not only raw rollout statistics
- **Phase progression**: curriculum advancement is gated by evaluation performance
- **Evaluation tooling**: random eval, checkpoint match eval, and league report CLI flows

---

## Environment

Each episode runs on a procedurally generated map with random internal wall placement while keeping both tanks reachable.

**Action space (`6` actions):**

- `NOOP`
- `LEFT`
- `RIGHT`
- `FWD`
- `BWD`
- `SHOOT`

**Combat and game rules:**

- Tanks move simultaneously
- Tanks can rotate, move, and shoot
- Shots are ray-based and blocked by walls
- Shooting uses cooldown steps
- Episodes end on kill or max step limit

**Rewards:**

- `+1.0` for winning
- `-1.0` for losing
- `0.0` for a draw
- Small step penalty to encourage faster outcomes
- Extra penalty for non-finishing shots

**Observation design:**

- Wall-distance features
- Enemy directional features only when actually visible
- Own direction, position, cooldown, and step progress
- Recent shot cue for hidden-information play

---

## Project Structure

```bash
src/
├── agents/
│   ├── policy.py
│   └── scripted_baselines.py
├── env/
│   ├── tank_env.py
│   ├── render.py
│   ├── entities.py
│   └── map_gen.py
├── evaluation/
│   ├── checkpoint_match.py
│   ├── eval_match.py
│   └── league_report.py
├── training/
│   ├── buffer.py
│   ├── ppo.py
│   └── self_play.py
└── scripts/
    ├── eval.py
    ├── league_report.py
    ├── match_eval.py
    ├── train.py
    └── models/
```

Key files:

- `src/env/tank_env.py`: hidden-information symmetric tank environment
- `src/agents/policy.py`: actor-critic policy network
- `src/training/self_play.py`: self-play training loop, pool logic, checkpoint policy, and progression
- `src/evaluation/eval_match.py`: random and saved-model evaluation
- `src/evaluation/checkpoint_match.py`: checkpoint-vs-checkpoint evaluation
- `src/evaluation/league_report.py`: compact league report across meaningful opponents

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **pip**

---

### Installation

```bash
# Clone the repository
git clone https://github.com/iPelo/Tank-Reinforcement-Learning.git

# Enter the project directory
cd Tank-Reinforcement-Learning

# Create a virtual environment (recommended)
python -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

Train the current hidden-information self-play setup with:

```bash
python -m src.scripts.train --phase 2 --single-phase
```

Run curriculum training across phases:

```bash
python -m src.scripts.train
```

Train against a fixed frozen opponent:

```bash
python -m src.scripts.train \
  --phase 2 \
  --single-phase \
  --opponent-checkpoint src/scripts/models/ppo_phase2_best.pt
```

Useful training controls:

- `--pool-opponent-prob`: probability of sampling a frozen opponent from the pool
- `--self-play-prob`: explicit probability for current-policy self-play
- `--snapshot-interval`: how often pool snapshots are created
- `--max-pool-size`: maximum retained pool snapshots per phase
- `--keep-every`: retain only snapshots aligned to a step interval
- `--eval-interval`: how often best-model evaluation runs
- `--best-eval-episodes`: episodes used for best-model evaluation
- `--promotion-eval-episodes`: episodes used for phase-promotion evaluation

Training logs include:

- update number
- environment steps
- sample count
- recent returns
- player/enemy win rates
- self-play vs frozen-opponent mix rates
- current opponent label
- PPO losses and KL

---

## Evaluation

Run a quick hidden-information eval:

```bash
python -m src.scripts.eval --phase 2 --episodes 1
```

Render hidden-information matches:

```bash
python -m src.scripts.eval --phase 2 --episodes 10 --render
```

Evaluate one checkpoint against another:

```bash
python -m src.scripts.match_eval \
  --player-model src/scripts/models/ppo_phase2_last.pt \
  --enemy-model src/scripts/models/ppo_phase2_best.pt \
  --phase 2 \
  --episodes 10
```

Generate a league report for the current checkpoint:

```bash
python -m src.scripts.league_report \
  --current-model src/scripts/models/ppo_phase2_last.pt \
  --phase 2 \
  --episodes 12
```

---

## Checkpoints and Opponent Pool

Training checkpoints are stored under `src/scripts/models/`.

Common files:

- `ppo_phase0_last.pt`
- `ppo_phase0_best.pt`
- `ppo_phase1_last.pt`
- `ppo_phase1_best.pt`
- `ppo_phase2_last.pt`
- `ppo_phase2_best.pt`

Opponent pool snapshots are stored by phase:

```bash
src/scripts/models/opponent_pool/
└── phase_2/
    ├── ppo_phase2_upd000010.pt
    ├── ppo_phase2_upd000020.pt
    └── ...
```

`best` checkpoints are now chosen through periodic evaluation, not only raw rollout performance.

---

## Roadmap Status

- **Phase 1**: symmetric `AI vs AI` self-play foundation
- **Phase 2**: hidden-information environment observations
- **Phase 3**: frozen opponents, opponent pool, checkpoint match eval, league metrics
- **Phase 4**: opponent-mix control, retention policy, eval-based best tracking, league reporting, eval-gated promotion
- **Next**: memory/recurrent policies, stronger league evaluation, and multi-tank combat

---

## Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **Pygame**

---

## Author

Built by [iPelo](https://github.com/iPelo)
