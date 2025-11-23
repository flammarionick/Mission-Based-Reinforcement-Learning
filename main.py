# main.py

import os
import sys
import argparse
import torch

from stable_baselines3 import PPO, A2C, DQN

from environment.custom_env import SmartBuildingSecurityEnv
from training.pg_training import PolicyNetwork  # for REINFORCE


def load_sb3_model(algo: str, model_path: str):
    """
    Load a Stable-Baselines3 model (DQN, PPO, A2C) from a .zip file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "a2c":
        model = A2C.load(model_path)
    elif algo == "dqn":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unsupported SB3 algo: {algo}")

    return model


def load_reinforce_policy(model_path: str, obs_dim: int, n_actions: int):
    """
    Load the custom REINFORCE policy network saved as a .pt file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"REINFORCE model path does not exist: {model_path}")

    policy = PolicyNetwork(obs_dim, n_actions)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_episode_sb3(algo: str, model_path: str, render: bool = True, max_steps: int = 500):
    """
    Run a single episode using an SB3 model (DQN, PPO, A2C).
    """
    env = SmartBuildingSecurityEnv()
    model = load_sb3_model(algo, model_path)

    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not done and not truncated and steps < max_steps:
        if render:
            env.render()

        # SB3 expects obs as array, returns action and state info
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_reward += reward
        steps += 1
        done = terminated

    env.close()
    print(f"[{algo.upper()}] Episode finished in {steps} steps, total reward = {total_reward:.2f}")


def run_episode_reinforce(model_path: str, render: bool = True, max_steps: int = 500):
    """
    Run a single episode using the REINFORCE policy network.
    """
    env = SmartBuildingSecurityEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = load_reinforce_policy(model_path, obs_dim, n_actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not done and not truncated and steps < max_steps:
        if render:
            env.render()

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        dist = policy(obs_tensor)
        action = dist.sample().item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated

    env.close()
    print(f"[REINFORCE] Episode finished in {steps} steps, total reward = {total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run trained RL agents (DQN / PPO / A2C / REINFORCE) in the Smart Building Security environment."
    )
    parser.add_argument(
        "algo",
        choices=["dqn", "ppo", "a2c", "reinforce"],
        help="Which algorithm's model to run."
    )
    parser.add_argument(
        "model_path",
        help="Path to the trained model file (.zip for SB3, .pt for REINFORCE)."
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable pygame rendering (for headless evaluation)."
    )
    args = parser.parse_args()

    render = not args.no_render

    if args.algo in ["dqn", "ppo", "a2c"]:
        run_episode_sb3(args.algo, args.model_path, render=render)
    else:
        # REINFORCE
        run_episode_reinforce(args.model_path, render=render)


if __name__ == "__main__":
    main()