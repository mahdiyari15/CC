# 🚗 Learning to Drive in CARLA — Imitation Learning & Reinforcement Learning

End-to-end pipeline for training autonomous-driving agents in the **CARLA** simulator: collect expert demonstrations, train a **behaviour-cloning** policy, then improve it with **offline RL (AWAC)** and online **Soft Actor-Critic (SAC)**.

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/CARLA-Simulator-1F2937?style=flat">
  <img src="https://img.shields.io/badge/Gymnasium-0081A5?style=flat">
</p>

## Overview

```
record demos (autopilot / manual)  ──►  behaviour cloning  ──►  offline RL (AWAC)  ──►  online RL (SAC)
                                                   ▲                                         │
                                                   └──────── closed-loop evaluation in CARLA ◄┘
```

- **Custom Gymnasium environment** (`CarlaEnv/`) on custom OpenDRIVE maps, with ego/vehicle/pedestrian spawners and pluggable observation adaptors (lane angle, object speeds, traffic signs).
- **Observations:** a bird's-eye semantic grid plus scalar driving features, encoded by a shared CNN feature extractor (`networks/`).
- **Actions:** continuous (steer, throttle/brake) or discrete control.
- **Imitation learning** (`imitation/`): demo recording, dataset building, BC training and evaluation with experiment logging.
- **Reinforcement learning** (`rl/`, `agents/`): SAC with replay buffer and evaluation scripts; AWAC-style offline training initialised from the BC policy.

## Repository layout

| Path | Contents |
|---|---|
| `CarlaEnv/` | Gymnasium wrapper around CARLA, maps (`.xodr`), spawners, observation adaptors, vehicle control |
| `imitation/` | Recording demonstrations, building datasets, training/evaluating BC |
| `rl/sac/` | SAC training, replay buffer, evaluation |
| `agents/` | BC and SAC agent classes |
| `networks/` | Feature extractor, actor and critic heads |
| `config/` | Hyper-parameters for BC, offline RL and SAC |
| `main/` | Entry points for the full pipelines |

## Running

Requires a running CARLA server (0.9.x) and its Python API.

```bash
# imitation-learning pipeline (record → dataset → train → evaluate)
python -m main.run_imitation

# reinforcement-learning pipeline (SAC)
python -m main.run_rl
```

Hyper-parameters live in `config/`.

---
Part of the **UT-Car** autonomous-driving project at the University of Tehran.
