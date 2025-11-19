# training/dqn_training.py

import os
from itertools import product

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import SmartBuildingSecurityEnv


def make_env():
    def _init():
        return SmartBuildingSecurityEnv()

    return _init


def train_dqn():
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)

    # Example hyperparameter grid – extend to reach >=10 runs
    learning_rates = [1e-4, 3e-4]
    gammas = [0.95, 0.99]
    batch_sizes = [64, 128]
    exploration_fractions = [0.1, 0.3]

    configs = list(
        product(learning_rates, gammas, batch_sizes, exploration_fractions)
    )

    total_timesteps = 150_000  # adjust as needed

    for idx, (lr, gamma, batch_size, exploration_fraction) in enumerate(configs):
        run_name = f"dqn_lr{lr}_g{gamma}_bs{batch_size}_ef{exploration_fraction}_run{idx}"

        print(f"\n=== Training DQN run {idx+1}/{len(configs)}: {run_name} ===")

        env = DummyVecEnv([make_env])

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            verbose=1,
            tensorboard_log="logs/dqn",
        )

        model.learn(total_timesteps=total_timesteps)
        model.save(os.path.join("models/dqn", f"{run_name}.zip"))

        env.close()


if __name__ == "__main__":
    train_dqn()