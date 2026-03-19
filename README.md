<h1 align="center">
  <a href="https://github.com/iPelo/Tank-Reinforcement-Learning">
    TANK-REINFORCEMENT-LEARNING
  </a>
</h1>

<p align="center">Train a PPO agent to battle in a custom grid-based tank environment</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/github/languages/top/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Top Language">
  <img src="https://img.shields.io/github/languages/count/iPelo/Tank-Reinforcement-Learning?style=for-the-badge" alt="Language Count">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Markdown-000000?logo=markdown&logoColor=white&style=for-the-badge" alt="Markdown">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge" alt="PyTorch">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge" alt="NumPy">
  <img src="https://img.shields.io/badge/Pygame-0E1117?logo=pygame&logoColor=white&style=for-the-badge" alt="Pygame">
</p>

---

## Table of Contents
- [Overview](#overview)
- [Environment](#environment)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)

---

## Overview

**Tank-Reinforcement-Learning** is a custom reinforcement learning project where an agent learns to fight an enemy tank in a **15x15 grid world** using **Proximal Policy Optimization (PPO)**.

The project includes:

- A custom `TankEnv` environment with walls, movement, turning, and shooting
- A PPO implementation with **GAE**, clipped policy loss, value loss, and entropy bonus
- A PyTorch **actor-critic** network for policy and value estimation
- Optional **Pygame rendering** for watching random or trained agents play

The agent is trained through increasingly difficult phases:

- `phase 0`: movement and positioning
- `phase 1`: combat with shooting enabled
- `phase 2`: full training setup used by default

---

## Environment

Each episode takes place in a procedurally generated map with random wall placement while keeping both tanks reachable.

**Action space (`6` actions):**

- `NOOP`
- `LEFT`
- `RIGHT`
- `FWD`
- `BWD`
- `SHOOT`

**Rewards:**

- `+1.0` for winning
- `-1.0` for losing
- `0.0` for a draw
- Small step penalty to encourage faster outcomes
- Extra penalty for shooting without ending the fight

Episodes end when:

- One tank is destroyed, or
- The maximum step limit is reached

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
│   ├── eval_match.py
│   └── watch.py
├── training/
│   ├── buffer.py
│   └── ppo.py
└── scripts/
    ├── train.py
    ├── eval.py
    ├── watch.py
    └── models/
```

Key files:

- `src/env/tank_env.py`: custom tank battle environment
- `src/agents/policy.py`: actor-critic policy network
- `src/training/buffer.py`: rollout buffer and GAE computation
- `src/training/ppo.py`: PPO update logic
- `src/scripts/train.py`: training entry point
- `src/evaluation/eval_match.py`: evaluation and rendering implementation
- `src/scripts/eval.py`: thin evaluation entry point

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

# Install dependencies
pip install -r requirements.txt
```

---

## Training

Train the PPO agent with:

```bash
python -m src.scripts.train --phase 2
```

You can also choose a different curriculum phase:

```bash
python -m src.scripts.train --phase 0
python -m src.scripts.train --phase 1
python -m src.scripts.train --phase 2
```

During training, the script logs:

- update number
- total environment steps
- recent mean return
- win/loss/draw rates
- PPO losses and approximate KL

---

## Evaluation

Run random-agent evaluation:

```bash
python -m src.scripts.eval --episodes 10 --phase 2
```

Run a trained model:

```bash
python -m src.scripts.eval --episodes 10 --phase 2 --model src/scripts/models/ppo_phase2_best.pt
```

Render gameplay with Pygame:

```bash
python -m src.scripts.eval --episodes 10 --phase 2 --model src/scripts/models/ppo_phase2_best.pt --render
```

---

## Checkpoints

Training automatically saves checkpoints under `src/scripts/models/`.

Files include:

- `ppo_phase0_last.pt`
- `ppo_phase0_best.pt`
- `ppo_phase1_last.pt`
- `ppo_phase1_best.pt`
- `ppo_phase2_last.pt`
- `ppo_phase2_best.pt`

`best` checkpoints are updated when the rolling win rate improves.

---

## Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **Pygame**

---

## Author

Built by [iPelo](https://github.com/iPelo)
