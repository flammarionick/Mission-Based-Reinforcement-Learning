# plot_cumulative_rewards.py

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import SmartBuildingSecurityEnv
from training.pg_training import PolicyNetwork  # REINFORCE network


N_EPISODES = 50  # number of eval episodes per algorithm


def eval_sb3_model(model_cls, model_path: str, algo_name: str):
    """
    Evaluate a Stable-Baselines3 model (DQN, PPO, A2C).
    Returns list of episode returns.
    """
    env = SmartBuildingSecurityEnv()
    model = model_cls.load(model_path)

    episode_returns = []

    for _ in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated

        episode_returns.append(total_reward)

    env.close()
    return np.array(episode_returns, dtype=float)


def eval_reinforce_model(model_path: str):
    """
    Evaluate the custom REINFORCE policy saved as reinforce_final.pt.
    Returns list of episode returns.
    """
    if not os.path.exists(model_path):
        print(f"[REINFORCE] Model file not found: {model_path}")
        return None

    env = SmartBuildingSecurityEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetwork(obs_dim, n_actions)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    episode_returns = []

    for _ in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            dist = policy(obs_tensor)
            action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated

        episode_returns.append(total_reward)

    env.close()
    return np.array(episode_returns, dtype=float)


def main():
    # ✅ Use the BEST models from your latest evaluate_all.py run:
    best_paths = {
        "DQN": os.path.join("models", "dqn", "dqn_lr0.0003_g0.95_bs128_ef0.1_run10.zip"),
        "A2C": os.path.join("models", "pg", "a2c_lr0.0001_g0.95_run4.zip"),
        "PPO": os.path.join("models", "pg", "ppo_lr0.0003_g0.99_ns256_run2.zip"),
        "REINFORCE": os.path.join("models", "pg", "reinforce_final.pt"),
    }

    # Evaluate each algorithm
    dqn_returns = eval_sb3_model(DQN, best_paths["DQN"], "dqn")
    a2c_returns = eval_sb3_model(A2C, best_paths["A2C"], "a2c")
    ppo_returns = eval_sb3_model(PPO, best_paths["PPO"], "ppo")
    reinforce_returns = eval_reinforce_model(best_paths["REINFORCE"])

    # Compute cumulative sums
    dqn_cum = np.cumsum(dqn_returns)
    a2c_cum = np.cumsum(a2c_returns)
    ppo_cum = np.cumsum(ppo_returns)
    rein_cum = np.cumsum(reinforce_returns) if reinforce_returns is not None else None

    episodes = np.arange(1, N_EPISODES + 1)

    # Plot 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Cumulative Rewards over Evaluation Episodes (Best Models)", fontsize=14)

    # DQN
    ax = axes[0, 0]
    ax.plot(episodes, dqn_cum)
    ax.set_title("DQN (best)")
    ax.set_ylabel("Cumulative Reward")

    # A2C
    ax = axes[0, 1]
    ax.plot(episodes, a2c_cum)
    ax.set_title("A2C (best)")

    # PPO
    ax = axes[1, 0]
    ax.plot(episodes, ppo_cum)
    ax.set_title("PPO (best)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")

    # REINFORCE
    ax = axes[1, 1]
    if rein_cum is not None:
        ax.plot(episodes, rein_cum)
        ax.set_title("REINFORCE")
    else:
        ax.text(0.5, 0.5, "REINFORCE model not found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("REINFORCE")

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots_cumulative_rewards.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()