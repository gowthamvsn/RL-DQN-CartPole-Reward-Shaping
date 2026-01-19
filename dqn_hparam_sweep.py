# ============================================================
# DQN CARTPOLE — HYPERPARAMETER SWEEP
# ============================================================

from dqn_reward_shaping import run_experiment

def main():
    configs = ["env"]
    for mode in configs:
        print(f"Running baseline for hyperparameter reference: {mode}")
        run_experiment(mode)

if __name__ == "__main__":
    main()
