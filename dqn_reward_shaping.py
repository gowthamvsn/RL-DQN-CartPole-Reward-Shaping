# ============================================================
# DQN CARTPOLE-V1 — REWARD SHAPING STUDY
# ============================================================

import random
from collections import deque, namedtuple
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
ENV_ID = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 10_000
TARGET_UPDATE_STEPS = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MAX_EPISODES = 800
SOLVED_THRESHOLD = 195.0
MOVING_AVG_WINDOW = 100
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))

# ==============================
# REPLAY BUFFER
# ==============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ==============================
# DQN MODEL
# ==============================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# REWARD FUNCTIONS
# ==============================
def reward_env(r, s, ns):
    return r

def reward_angle(r, s, ns):
    return -abs(ns[2])

def reward_angle_pos(r, s, ns):
    return -(abs(ns[2]) + abs(ns[0]))

REWARDS = {
    "env": reward_env,
    "angle": reward_angle,
    "angle_pos": reward_angle_pos
}

# ==============================
# TRAINING
# ==============================
def run_experiment(mode):
    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n

    policy = DQN(n_obs, n_act).to(DEVICE)
    target = DQN(n_obs, n_act).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    memory = ReplayBuffer(REPLAY_CAPACITY)

    eps = EPS_START
    rewards, moving_avg = [], []
    global_step = 0

    for ep in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        ep_return = 0.0

        for _ in count():
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            if random.random() < eps:
                action = random.randrange(n_act)
            else:
                with torch.no_grad():
                    action = policy(state_t).argmax().item()

            next_state, env_r, term, trunc, _ = env.step(action)
            done = term or trunc
            shaped = REWARDS[mode](env_r, state, next_state)

            memory.push(state, action, shaped, next_state, done)
            state = next_state
            ep_return += env_r

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                s = torch.tensor(batch.state, dtype=torch.float32, device=DEVICE)
                a = torch.tensor(batch.action, device=DEVICE).unsqueeze(1)
                r = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
                ns = torch.tensor(batch.next_state, dtype=torch.float32, device=DEVICE)
                d = torch.tensor(batch.done, dtype=torch.bool, device=DEVICE)

                q = policy(s).gather(1, a).squeeze()
                with torch.no_grad():
                    nq = target(ns).max(1)[0]
                    nq[d] = 0.0
                loss = nn.MSELoss()(q, r + GAMMA * nq)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            global_step += 1
            if global_step % TARGET_UPDATE_STEPS == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                break

        rewards.append(ep_return)
        eps = max(EPS_END, eps * EPS_DECAY)
        avg = np.mean(rewards[-MOVING_AVG_WINDOW:])
        moving_avg.append(avg)

        print(f"[{mode}] ep={ep:3d} return={ep_return:.1f} avg100={avg:.1f}")

        if len(rewards) >= MOVING_AVG_WINDOW and avg >= SOLVED_THRESHOLD:
            break

    env.close()
    return rewards, moving_avg

# ==============================
# MAIN
# ==============================
def main():
    results = {}
    for mode in REWARDS:
        results[mode] = run_experiment(mode)

    for mode, (r, _) in results.items():
        plt.plot(r, label=mode)
    plt.legend(); plt.title("Episode Return"); plt.show()

if __name__ == "__main__":
    main()
