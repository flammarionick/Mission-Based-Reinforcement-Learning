import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Helper to load a scalar time-series from a TB log directory
def load_scalars(log_dir, tag):
    if not os.path.isdir(log_dir):
        print(f"[WARN] log dir not found: {log_dir}")
        return None, None

    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        print(f"[WARN] tag '{tag}' not found in {log_dir}. Available: {tags}")
        return None, None

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Stability: DQN Loss and Policy Entropy (PPO, A2C)")

    # ----- DQN: TD loss (objective function) -----
    dqn_steps, dqn_loss = load_scalars("logs/dqn", "train/loss")
    ax = axes[0]
    if dqn_loss:
        ax.plot(dqn_steps, dqn_loss)
        ax.set_title("DQN TD-Loss")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No DQN loss logs found", ha="center")

    # ----- PPO: entropy loss (policy entropy proxy) -----
    ppo_steps, ppo_entropy = load_scalars("logs/ppo", "train/entropy_loss")
    ax = axes[1]
    if ppo_entropy:
        ax.plot(ppo_steps, ppo_entropy)
        ax.set_title("PPO Policy Entropy")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No PPO entropy logs found", ha="center")

    # ----- A2C: entropy loss -----
    a2c_steps, a2c_entropy = load_scalars("logs/a2c", "train/entropy_loss")
    ax = axes[2]
    if a2c_entropy:
        ax.plot(a2c_steps, a2c_entropy)
        ax.set_title("A2C Policy Entropy")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No A2C entropy logs found", ha="center")

    plt.tight_layout()
    plt.savefig("fig_training_stability.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()