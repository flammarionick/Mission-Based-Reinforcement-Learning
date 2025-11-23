# training/evaluate_models.py

import os
import csv
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO, DQN, A2C

from environment.custom_env import SmartBuildingSecurityEnv


def evaluate_sb3_model(
    model,
    n_episodes: int = 20,
) -> Tuple[float, float, float]:
    """
    Run the model for n_episodes and return:
      - mean_reward
      - std_reward
      - mean_episode_length
    """
    env = SmartBuildingSecurityEnv()
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            ep_len += 1
            done = terminated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    env.close()

    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    mean_len = float(np.mean(episode_lengths))
    return mean_r, std_r, mean_len


def evaluate_folder(
    folder: str,
    algo: str,
    output_csv: str,
    best_only: bool = False,
):
    """
    Evaluate all SB3 models in a folder and write metrics to CSV.

    If best_only=True, only evaluates 'best_model.zip' under *_best subfolders.
    """
    if algo.lower() == "ppo":
        ModelCls = PPO
    elif algo.lower() == "dqn":
        ModelCls = DQN
    elif algo.lower() == "a2c":
        ModelCls = A2C
    else:
        raise ValueError("Unsupported algo. Use: ppo, dqn, a2c")

    rows = []

    if best_only:
        # Look for paths like <folder>/<run_name>_best/best_model.zip
        for sub in os.listdir(folder):
            sub_path = os.path.join(folder, sub)
            if not os.path.isdir(sub_path):
                continue
            best_path = os.path.join(sub_path, "best_model.zip")
            if os.path.exists(best_path):
                print(f"Evaluating {best_path} ...")
                model = ModelCls.load(best_path)
                mean_r, std_r, mean_len = evaluate_sb3_model(model)
                rows.append(
                    [sub, best_path, mean_r, std_r, mean_len]
                )
    else:
        # Evaluate every .zip in folder
        for fname in os.listdir(folder):
            if not fname.endswith(".zip"):
                continue
            model_path = os.path.join(folder, fname)
            print(f"Evaluating {model_path} ...")
            model = ModelCls.load(model_path)
            mean_r, std_r, mean_len = evaluate_sb3_model(model)
            rows.append(
                [fname, model_path, mean_r, std_r, mean_len]
            )

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["run_name", "model_path", "mean_reward", "std_reward", "mean_episode_len"]
        )
        writer.writerows(rows)

    print(f"Saved evaluation metrics to {output_csv}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python -m training.evaluate_models <algo> <models_folder> <output_csv> [best_only]")
        print('Example: python -m training.evaluate_models ppo models/pg "results/ppo_eval.csv" best')
        sys.exit(0)

    algo = sys.argv[1].lower()
    models_folder = sys.argv[2]
    output_csv = sys.argv[3]
    best_only_flag = False
    if len(sys.argv) >= 5 and sys.argv[4].lower().startswith("best"):
        best_only_flag = True

    evaluate_folder(models_folder, algo, output_csv, best_only=best_only_flag)