# main.py
"""
Run a trained RL agent (DQN / PPO / A2C / REINFORCE) in the
SmartBuildingSecurityEnv with live visualization and verbose logs.

Usage examples:

  # PPO best model
  python main.py ppo models/pg/ppo_lr0.0003_g0.99_ns256_run2.zip

  # A2C best model
  python main.py a2c models/pg/a2c_lr0.0001_g0.95_run4.zip

  # DQN best model
  python main.py dqn models/dqn/dqn_lr0.0003_g0.95_bs128_ef0.1_run10.zip

  # REINFORCE (defaults to models/pg/reinforce_final.pt)
  python main.py reinforce

You can also add:

  --episodes 3        # number of episodes to roll out
  --max-steps 100     # safety cap on steps per episode
"""

import os
import argparse
import time

import numpy as np
import torch

from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import SmartBuildingSecurityEnv
from training.pg_training import PolicyNetwork  # for REINFORCE


def make_env():
    """Create a fresh environment instance."""
    # If your env accepts render_mode="human", you can pass it here.
    # For now we just call the default constructor.
    return SmartBuildingSecurityEnv()


# ---------------------- SB3 AGENT ROLLOUT ---------------------- #

def run_sb3_agent(algo: str, model_path: str, episodes: int = 1, max_steps: int = 50):
    """Run a trained SB3 agent (DQN/PPO/A2C) with rendering + verbose logs."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    algo = algo.lower()
    if algo == "dqn":
        ModelCls = DQN
    elif algo == "ppo":
        ModelCls = PPO
    elif algo == "a2c":
        ModelCls = A2C
    else:
        raise ValueError(f"Unsupported SB3 algorithm: {algo}")

    print(f"[{algo.upper()}] Loading model from: {model_path}")
    model = ModelCls.load(model_path)

    env = make_env()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step_idx = 0

        # Initial render
        env.render()

        print(f"\n[{algo.upper()}] === Episode {ep + 1}/{episodes} ===")
        while not done and not truncated and step_idx < max_steps:
            step_idx += 1

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward

            # Verbose logging
            print(
                f"[{algo.upper()}] step={step_idx:02d} | "
                f"action={int(action)} | reward={reward:+.2f} | "
                f"cum_reward={total_reward:+.2f} | done={bool(terminated)}"
            )

            # Render environment (Pygame)
            env.render()

            # small sleep so it is human-readable in the video
            time.sleep(0.15)

            obs = next_obs
            done = terminated

        print(
            f"[{algo.upper()}] Episode {ep + 1} finished in {step_idx} steps, "
            f"total reward = {total_reward:.2f}"
        )

    env.close()


# ---------------------- REINFORCE AGENT ROLLOUT ---------------------- #

def run_reinforce_agent(model_path: str, episodes: int = 1, max_steps: int = 50):
    """Run the custom REINFORCE policy (PyTorch) with rendering + verbose logs."""
    if model_path is None:
        model_path = os.path.join("models", "pg", "reinforce_final.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"REINFORCE model file not found: {model_path}")

    print(f"[REINFORCE] Loading policy from: {model_path}")

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step_idx = 0

        env.render()

        print(f"\n[REINFORCE] === Episode {ep + 1}/{episodes} ===")
        while not done and not truncated and step_idx < max_steps:
            step_idx += 1

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            dist = policy(obs_tensor)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"[REINFORCE] step={step_idx:02d} | "
                f"action={action} | reward={reward:+.2f} | "
                f"cum_reward={total_reward:+.2f} | done={bool(terminated)}"
            )

            env.render()
            time.sleep(0.15)

            obs = next_obs
            done = terminated

        print(
            f"[REINFORCE] Episode {ep + 1} finished in {step_idx} steps, "
            f"total reward = {total_reward:.2f}"
        )

    env.close()


# ---------------------- CLI ENTRY POINT ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Run trained RL agents in the SmartBuildingSecurityEnv."
    )
    parser.add_argument(
        "algo",
        type=str,
        choices=["dqn", "ppo", "a2c", "reinforce"],
        help="Which algorithm to run.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to model file (.zip for SB3, .pt for REINFORCE). "
             "For REINFORCE, defaults to models/pg/reinforce_final.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to roll out (default: 1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Safety cap on steps per episode (default: 50).",
    )

    args = parser.parse_args()

    if args.algo == "reinforce":
        run_reinforce_agent(
            model_path=args.model_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
    else:
        if args.model_path is None:
            raise ValueError("You must specify model_path for DQN/PPO/A2C.")
        run_sb3_agent(
            algo=args.algo,
            model_path=args.model_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()