# training/pg_training.py

import os
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import SmartBuildingSecurityEnv


# ------------- Helper: Environment factory ------------- #

def make_env():
    """
    Factory for SmartBuildingSecurityEnv (for DummyVecEnv).
    DummyVecEnv expects a callable that returns an Env instance.
    """
    return SmartBuildingSecurityEnv()


# ------------- PPO and A2C training using SB3 ------------- #

def train_ppo():
    """
    Train multiple PPO agents on the SmartBuildingSecurityEnv
    with an expanded hyperparameter grid and EvalCallback.
    """
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)
    os.makedirs("results/ppo", exist_ok=True)

    # Expanded grid: 3 x 2 x 2 = 12 runs
    learning_rates = [3e-4, 1e-4, 5e-5]
    gammas = [0.95, 0.99]
    n_steps_list = [256, 512]

    configs = list(product(learning_rates, gammas, n_steps_list))
    total_timesteps = 200_000

    for idx, (lr, gamma, n_steps) in enumerate(configs):
        run_name = f"ppo_lr{lr}_g{gamma}_ns{n_steps}_run{idx}"
        print(f"\n=== Training PPO run {idx + 1}/{len(configs)}: {run_name} ===")

        env = DummyVecEnv([make_env])
        eval_env = DummyVecEnv([make_env])

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join("models/pg", f"{run_name}_best"),
            log_path=os.path.join("results/ppo", run_name),
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=n_steps,
            verbose=1,
            tensorboard_log="logs/ppo",
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(os.path.join("models/pg", f"{run_name}.zip"))

        env.close()
        eval_env.close()


def train_a2c():
    """
    Train multiple A2C agents on the SmartBuildingSecurityEnv
    with an expanded hyperparameter grid and EvalCallback.
    """
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("logs/a2c", exist_ok=True)
    os.makedirs("results/a2c", exist_ok=True)

    # Expanded grid: 5 x 2 = 10 runs
    learning_rates = [7e-4, 3e-4, 1e-4, 5e-4, 2e-4]
    gammas = [0.95, 0.99]

    configs = list(product(learning_rates, gammas))
    total_timesteps = 200_000

    for idx, (lr, gamma) in enumerate(configs):
        run_name = f"a2c_lr{lr}_g{gamma}_run{idx}"
        print(f"\n=== Training A2C run {idx + 1}/{len(configs)}: {run_name} ===")

        env = DummyVecEnv([make_env])
        eval_env = DummyVecEnv([make_env])

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join("models/pg", f"{run_name}_best"),
            log_path=os.path.join("results/a2c", run_name),
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            verbose=1,
            tensorboard_log="logs/a2c",
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(os.path.join("models/pg", f"{run_name}.zip"))

        env.close()
        eval_env.close()


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
    """
    Vanilla REINFORCE implementation on the SmartBuildingSecurityEnv.
    """
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
        print(f"[REINFORCE] Episode {ep + 1}/{episodes}, Return: {ep_return:.2f}")

    torch.save(policy.state_dict(), save_path)
    env.close()
    print(f"Saved REINFORCE policy to {save_path}")


if __name__ == "__main__":
    # Example usage:
    #   python -m training.pg_training ppo
    #   python -m training.pg_training a2c
    #   python -m training.pg_training reinforce

    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m training.pg_training [ppo|a2c|reinforce]")
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