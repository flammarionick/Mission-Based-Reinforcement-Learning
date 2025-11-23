"""
Evaluate all trained models (DQN, PPO, A2C, REINFORCE) on the
SmartBuildingSecurityEnv and save summary CSVs for the report.

Usage:
    python evaluate_all.py

This will produce:
    - ppo_eval.csv
    - a2c_eval.csv
    - dqn_eval.csv
    - reinforce_eval.csv (if reinforce_final.pt exists)

and print best model per algorithm.
"""

import os
import csv
import numpy as np

from stable_baselines3 import PPO, A2C, DQN

from environment.custom_env import SmartBuildingSecurityEnv
from training.pg_training import PolicyNetwork  # for REINFORCE

import torch


N_EPISODES = 10  # number of evaluation episodes per model


def evaluate_sb3_model(model_cls, model_path: str, algo_name: str):
    """
    Evaluate a Stable-Baselines3 model (PPO, A2C, DQN) for N_EPISODES.
    Returns mean_reward, std_reward, mean_ep_len.
    """
    env = SmartBuildingSecurityEnv()
    model = model_cls.load(model_path)

    episode_rewards = []
    episode_lengths = []

    for _ in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_ep_len = float(np.mean(episode_lengths))

    print(
        f"[{algo_name.upper()}] {os.path.basename(model_path)} "
        f"=> mean_reward={mean_reward:.2f}, std={std_reward:.2f}, mean_len={mean_ep_len:.2f}"
    )

    return mean_reward, std_reward, mean_ep_len


def evaluate_reinforce_model(model_path: str):
    """
    Evaluate the custom REINFORCE policy saved as reinforce_final.pt.
    """
    if not os.path.exists(model_path):
        print(f"[REINFORCE] Model file not found: {model_path} (skipping)")
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

    episode_rewards = []
    episode_lengths = []

    for _ in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done and not truncated:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            dist = policy(obs_tensor)
            action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_ep_len = float(np.mean(episode_lengths))

    print(
        f"[REINFORCE] {os.path.basename(model_path)} "
        f"=> mean_reward={mean_reward:.2f}, std={std_reward:.2f}, mean_len={mean_ep_len:.2f}"
    )

    return mean_reward, std_reward, mean_ep_len


def eval_family(
    algo_name: str,
    model_dir: str,
    prefix: str,
    model_cls,
):
    """
    Evaluate a family of SB3 models (PPO, A2C, DQN) in a folder
    and write a CSV "<algo>_eval.csv".
    """
    csv_path = f"{algo_name}_eval.csv"
    rows = []

    if not os.path.isdir(model_dir):
        print(f"[{algo_name.upper()}] Directory not found: {model_dir} (skipping)")
        return None

    files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".zip")]
    files.sort()

    if not files:
        print(f"[{algo_name.upper()}] No models found in {model_dir} with prefix '{prefix}'")
        return None

    for fname in files:
        full_path = os.path.join(model_dir, fname)
        mean_r, std_r, mean_len = evaluate_sb3_model(model_cls, full_path, algo_name)

        rows.append(
            {
                "run_name": fname,
                "model_path": full_path,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "mean_episode_len": mean_len,
            }
        )

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "model_path", "mean_reward", "std_reward", "mean_episode_len"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # pick best
    best = max(rows, key=lambda r: r["mean_reward"])
    print(
        f"\n>>> BEST {algo_name.upper()} model: {best['run_name']} "
        f"(mean_reward={best['mean_reward']:.2f})\n"
    )

    return best


def main():
    # 1) Evaluate PPO (models/pg/ppo_*.zip)
    best_ppo = eval_family(
        algo_name="ppo",
        model_dir=os.path.join("models", "pg"),
        prefix="ppo_",
        model_cls=PPO,
    )

    # 2) Evaluate A2C (models/pg/a2c_*.zip)
    best_a2c = eval_family(
        algo_name="a2c",
        model_dir=os.path.join("models", "pg"),
        prefix="a2c_",
        model_cls=A2C,
    )

    # 3) Evaluate DQN (models/dqn/dqn_*.zip)
    best_dqn = eval_family(
        algo_name="dqn",
        model_dir=os.path.join("models", "dqn"),
        prefix="dqn_",
        model_cls=DQN,
    )

    # 4) Evaluate REINFORCE (if present)
    reinforce_path = os.path.join("models", "pg", "reinforce_final.pt")
    best_reinforce = evaluate_reinforce_model(reinforce_path)

    # Optionally, write a small summary text file with best models.
    with open("best_models_summary.txt", "w") as f:
        if best_ppo:
            f.write(
                f"BEST PPO: {best_ppo['run_name']} "
                f"(mean_reward={best_ppo['mean_reward']:.2f})\n"
            )
        if best_a2c:
            f.write(
                f"BEST A2C: {best_a2c['run_name']} "
                f"(mean_reward={best_a2c['mean_reward']:.2f})\n"
            )
        if best_dqn:
            f.write(
                f"BEST DQN: {best_dqn['run_name']} "
                f"(mean_reward={best_dqn['mean_reward']:.2f})\n"
            )
        if best_reinforce is not None:
            m, s, l = best_reinforce
            f.write(
                f"REINFORCE (reinforce_final.pt): "
                f"mean_reward={m:.2f}, std={s:.2f}, mean_len={l:.2f}\n"
            )


if __name__ == "__main__":
    main()