# main.py

import os
import sys
import time

import numpy as np
from stable_baselines3 import DQN, PPO, A2C
import torch

from environment.custom_env import SmartBuildingSecurityEnv
from environment.rendering import SecurityEnvRenderer
from training.pg_training import PolicyNetwork  # for REINFORCE policy


def run_sb3_agent(model_path, algo="dqn", episodes=5, fps=8):
    env = SmartBuildingSecurityEnv()
    renderer = SecurityEnvRenderer(env)

    # Load the model
    if algo.lower() == "dqn":
        model_cls = DQN
    elif algo.lower() == "ppo":
        model_cls = PPO
    elif algo.lower() == "a2c":
        model_cls = A2C
    else:
        raise ValueError("Unsupported SB3 algo for this runner.")

    model = model_cls.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            renderer.handle_events()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            info["reward"] = reward

            renderer.draw(info=info)
            renderer.clock.tick(fps)

            done = terminated

    renderer.close()
    env.close()


def run_reinforce_agent(model_path, episodes=5, fps=8):
    env = SmartBuildingSecurityEnv()
    renderer = SecurityEnvRenderer(env)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            renderer.handle_events()

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            dist = policy(obs_tensor)
            action = dist.probs.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            info["reward"] = reward

            renderer.draw(info=info)
            renderer.clock.tick(fps)

            done = terminated

    renderer.close()
    env.close()


if __name__ == "__main__":
    """
    Example usage:

      # Run best DQN model:
      python main.py dqn models/dqn/dqn_best.zip

      # Run best PPO model:
      python main.py ppo models/pg/ppo_best.zip

      # Run REINFORCE policy:
      python main.py reinforce models/pg/reinforce_final.pt
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py [dqn|ppo|a2c|reinforce] <model_path>")
        sys.exit(0)

    algo = sys.argv[1].lower()
    model_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        sys.exit(1)

    if algo in ["dqn", "ppo", "a2c"]:
        run_sb3_agent(model_path, algo=algo)
    elif algo == "reinforce":
        run_reinforce_agent(model_path)
    else:
        print("Unknown algorithm. Use: dqn, ppo, a2c, reinforce")