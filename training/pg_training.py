# training/pg_training.py

import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import SmartBuildingSecurityEnv


# ------------- Helper: Environment factory ------------- #

def make_env():
    def _init():
        return SmartBuildingSecurityEnv()
    return _init


# ------------- PPO and A2C training using SB3 ------------- #

def train_ppo():
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)

    learning_rates = [3e-4, 1e-4]
    gammas = [0.95, 0.99]
    n_steps_list = [256, 512]

    configs = list(product(learning_rates, gammas, n_steps_list))
    total_timesteps = 200_000

    for idx, (lr, gamma, n_steps) in enumerate(configs):
        run_name = f"ppo_lr{lr}_g{gamma}_ns{n_steps}_run{idx}"
        print(f"\n=== Training PPO run {idx+1}/{len(configs)}: {run_name} ===")

        env = DummyVecEnv([make_env])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=n_steps,
            verbose=1,
            tensorboard_log="logs/ppo",
        )

        model.learn(total_timesteps=total_timesteps)
        model.save(os.path.join("models/pg", f"{run_name}.zip"))

        env.close()


def train_a2c():
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("logs/a2c", exist_ok=True)

    learning_rates = [7e-4, 3e-4]
    gammas = [0.95, 0.99]

    configs = list(product(learning_rates, gammas))
    total_timesteps = 200_000

    for idx, (lr, gamma) in enumerate(configs):
        run_name = f"a2c_lr{lr}_g{gamma}_run{idx}"
        print(f"\n=== Training A2C run {idx+1}/{len(configs)}: {run_name} ===")

        env = DummyVecEnv([make_env])

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            verbose=1,
            tensorboard_log="logs/a2c",
        )

        model.learn(total_timesteps=total_timesteps)
        model.save(os.path.join("models/pg", f"{run_name}.zip"))

        env.close()


# ------------- Simple REINFORCE implementation ------------- #

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


def train_reinforce(
    episodes=500,
    gamma=0.99,
    learning_rate=1e-3,
    hidden_sizes=(128, 128),
    save_path="models/pg/reinforce_final.pt",
):
    os.makedirs("models/pg", exist_ok=True)

    env = SmartBuildingSecurityEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetwork(obs_dim, n_actions, hidden_sizes).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        log_probs = []
        rewards = []

        while not done and not truncated:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            dist = policy(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, info = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs
            done = terminated

        # Compute returns (G_t)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # Normalize returns to stabilize training
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_tensor = torch.stack(log_probs)

        loss = -torch.sum(log_probs_tensor * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_return = sum(rewards)
        print(f"[REINFORCE] Episode {ep+1}/{episodes}, Return: {ep_return:.2f}")

    torch.save(policy.state_dict(), save_path)
    env.close()
    print(f"Saved REINFORCE policy to {save_path}")


if __name__ == "__main__":
    # You can call them separately when needed.
    # Example:
    #   python training/pg_training.py ppo
    #   python training/pg_training.py a2c
    #   python training/pg_training.py reinforce

    import sys

    if len(sys.argv) < 2:
        print("Usage: python training/pg_training.py [ppo|a2c|reinforce]")
        sys.exit(0)

    mode = sys.argv[1].lower()
    if mode == "ppo":
        train_ppo()
    elif mode == "a2c":
        train_a2c()
    elif mode == "reinforce":
        train_reinforce()
    else:
        print("Unknown mode. Use one of: ppo, a2c, reinforce")