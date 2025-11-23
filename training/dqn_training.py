# training/dqn_training.py

import os
from itertools import product

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import SmartBuildingSecurityEnv


def make_env():
    """Factory for SmartBuildingSecurityEnv (for DummyVecEnv)."""
    return SmartBuildingSecurityEnv()


def train_dqn():
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("results/dqn", exist_ok=True)

    # Expanded grid: 3 x 2 x 2 x 1 = 12 runs
    learning_rates = [1e-4, 3e-4, 1e-3]
    gammas = [0.95, 0.99]
    batch_sizes = [64, 128]
    exploration_fractions = [0.1, 0.3]

    configs = list(
        product(learning_rates, gammas, batch_sizes, exploration_fractions)
    )

    total_timesteps = 200_000  # adjust if needed

    for idx, (lr, gamma, batch_size, exploration_fraction) in enumerate(configs):
        run_name = (
            f"dqn_lr{lr}_g{gamma}_bs{batch_size}_ef{exploration_fraction}_run{idx}"
        )

        print(f"\n=== Training DQN run {idx+1}/{len(configs)}: {run_name} ===")

        # Training env
        env = DummyVecEnv([make_env])

        # Evaluation env (vectorized)
        eval_env = DummyVecEnv([make_env])

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join("models/dqn", f"{run_name}_best"),
            log_path=os.path.join("results/dqn", run_name),
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

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

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        # Save final model as well
        model.save(os.path.join("models/dqn", f"{run_name}.zip"))

        env.close()
        eval_env.close()


if __name__ == "__main__":
    train_dqn()