## Smart Building Security RL – Summative Project

This project implements a custom reinforcement learning environment for a **smart building security mission**, and compares **value-based** and **policy-gradient** methods using Stable Baselines3 and a custom REINFORCE implementation.

## Environment Overview

The `SmartBuildingSecurityEnv` simulates a building with multiple zones:

- The agent monitors zones, triggers alarms, and dispatches guards.
- Intruders occasionally appear and move towards critical assets.
- The agent is rewarded for early detection and penalised for breaches and false alarms.

## Project Structure

```text
project_root/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment
│   ├── rendering.py         # Pygame visualisation + random agent demo
├── training/
│   ├── dqn_training.py      # DQN training (value-based)
│   ├── pg_training.py       # PPO, A2C (SB3) + REINFORCE (custom)
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved PPO/A2C/REINFORCE models
├── main.py                  # Run best-performing agent with GUI
├── requirements.txt         # Dependencies
└── README.md
Setup
bash
Copy code
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
Running the Environment (Random Agent Demo)
bash
Copy code
python -m environment.rendering
Use a screen recorder while this is running to generate the static video of random actions in the custom environment.

Training
DQN (Value-Based)
bash
Copy code
python -m training.dqn_training
This runs a hyperparameter sweep and saves models in models/dqn/.

PPO, A2C, REINFORCE
bash
Copy code
# PPO
python -m training.pg_training ppo

# A2C
python -m training.pg_training a2c

# REINFORCE
python -m training.pg_training reinforce
Models are saved in models/pg/.

Running the Best Agent
Pick your best-performing model (e.g., models/dqn/dqn_lr...zip) and run:

bash
Copy code
python main.py dqn models/dqn/<your_best_model>.zip
For PPO/A2C:

bash
Copy code
python main.py ppo models/pg/<your_best_ppo_model>.zip
python main.py a2c models/pg/<your_best_a2c_model>.zip
For REINFORCE:

bash
Copy code
python main.py reinforce models/pg/reinforce_final.pt
This will open the pygame window and visualise the agent in real time.

yaml
Copy code

---

If you want, next we can:

- tighten the reward/termination logic to match exactly how you want to discuss it in the report, and
- sketch out how to log metrics (avg reward, success rate) into CSVs for your graphs.
