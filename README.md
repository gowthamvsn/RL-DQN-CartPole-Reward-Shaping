How does reward shaping affect learning dynamics, stability, and solvability in DQN?

# DQN on CartPole-v1: Reward Shaping & Hyperparameter Analysis

This project implements a Deep Q-Network (DQN) agent for the CartPole-v1
environment and studies how different reward formulations affect learning
dynamics, convergence speed, and stability.

The project consists of two controlled experiments:
1. Reward shaping comparison (main study)
2. Hyperparameter sensitivity analysis (secondary study)

---

## Why This Project?

Reward shaping is often used in reinforcement learning to accelerate learning,
but poorly designed rewards can destabilize training or prevent convergence.

This project investigates:
- When reward shaping helps
- When it hurts
- Why some shaped rewards fail despite intuitive appeal

The goal is to develop intuition about **value estimation stability** in DQN.

---
## Project structure 
```
DQN-CartPole-Reward-Shaping/
├── src/
│   ├── dqn_reward_shaping.py      # Main experiment (env vs angle vs angle_pos)
│   └── dqn_hparam_sweep.py        # Hyperparameter study
│
├── README.md
├── requirements.txt
└── .gitignore
```


## Environment

- **Environment**: CartPole-v1 (Gymnasium)
- **State**: [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions**: left / right
- **Success Criterion**:  
  Average reward ≥ 195 over the last 100 episodes

---

## Experiment 1: Reward Shaping (Main)

We compare three reward schemes using identical hyperparameters:

| Reward Mode | Learning Signal |
|-----------|----------------|
| `env` | Original environment reward |
| `angle` | -\|pole_angle\| |
| `angle_pos` | -(\|pole_angle\| + \|cart_position\|) |

The agent learns using the shaped reward, but success is always evaluated
using the original environment reward.

📌 **File**: `src/dqn_reward_shaping.py`

---

## Experiment 2: Hyperparameter Sweep

We evaluate how learning speed and stability change when varying:
- Learning rate
- Target network update frequency
- Batch size
- Epsilon decay

📌 **File**: `src/dqn_hparam_sweep.py`

---

## Key Findings (Summary)

- Original reward (`env`) converges reliably but slowly
- Angle-based shaping converges faster and more stably
- Angle + position shaping fails due to value estimation instability
- Faster target network updates improve convergence
- Higher learning rates degrade stability

---

## Reproducibility

- Framework: PyTorch
- RL Library: Gymnasium
- Device: CPU or CUDA (if available)
- Seed: 42
- Max episodes: 800 (early stopping on solve)

---
<img width="630" height="470" alt="dqn_pic1" src="https://github.com/user-attachments/assets/188e759b-bb1f-471c-a6d6-64fdb7e033e2" />
<img width="630" height="470" alt="dqn_pic2" src="https://github.com/user-attachments/assets/644b71c0-5c4c-4d10-9bdd-1cc64f784b4a" />
<img width="628" height="470" alt="dqn_pic3" src="https://github.com/user-attachments/assets/e10b50ba-d355-4097-81c9-02e7744b9024" />

## How to Run

```bash
pip install -r requirements.txt

python src/dqn_reward_shaping.py
python src/dqn_hparam_sweep.py

## Author
Gowtham Vuppaladhadiam
