# environment/custom_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SmartBuildingSecurityEnv(gym.Env):
    """
    Smart Building Security Environment

    The agent manages security across multiple zones in a building.
    It must detect intruders early while avoiding excessive false alarms
    and unnecessary guard dispatches.

    Zones example:
        0: Entrance
        1: Lobby
        2: Corridor
        3: Server Room (critical)

    Actions (Discrete):
        0 – Stay monitoring current zone
        1 – Switch focus to Zone 0
        2 – Switch focus to Zone 1
        3 – Switch focus to Zone 2
        4 – Switch focus to Zone 3
        5 – Trigger alarm in current zone
        6 – Dispatch guard to current zone

    Observation (Box):
        For each zone i:
            motion[i]          ∈ [0, 1]
            time_since_scan[i] ∈ [0, 1]
            is_critical[i]     ∈ [0, 1] (constant per zone)

        Global features:
            guard_busy         ∈ [0, 1]
            time_step_norm     ∈ [0, 1]
            intrusion_active   ∈ [0, 1]
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 10}

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {}

        self.n_zones = config.get("n_zones", 4)
        self.max_steps = config.get("max_steps", 50)
        self.intruder_spawn_prob = config.get("intruder_spawn_prob", 0.05)
        self.max_intruder_steps = config.get("max_intruder_steps", 10)
        self.guard_travel_time = config.get("guard_travel_time", 3)

        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)

        # Observation space
        # For each zone: motion, time_since_scan, is_critical
        zone_features = 3 * self.n_zones
        # Global: guard_busy, time_step_norm, intrusion_active
        global_features = 3
        obs_dim = zone_features + global_features

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Critical zones: last zone is critical by default
        self.critical_zones = np.zeros(self.n_zones, dtype=np.float32)
        self.critical_zones[-1] = 1.0

        # Internal state
        self.current_zone = 0
        self.time_step = 0
        self.motion = np.zeros(self.n_zones, dtype=np.float32)
        self.time_since_scan = np.zeros(self.n_zones, dtype=np.float32)

        self.intruder_active = False
        self.intruder_zone = None
        self.intruder_progress = 0.0  # 0 -> far, 1 -> reached asset

        self.guard_busy = False
        self.guard_zone = None
        self.guard_eta = 0

        self.last_action = None

    # ------------ Gym API ------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time_step = 0
        self.current_zone = 0
        self.motion[:] = 0.0
        self.time_since_scan[:] = 0.0

        self.intruder_active = False
        self.intruder_zone = None
        self.intruder_progress = 0.0

        self.guard_busy = False
        self.guard_zone = None
        self.guard_eta = 0

        self.last_action = None

        # Optionally spawn an intruder at reset
        if self.np_random.random() < 0.5:
            self._spawn_intruder()

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        self.last_action = int(action)
        self.time_step += 1

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Small step penalty to encourage efficiency
        reward -= 0.1

        # 1. Apply the chosen action
        if action == 0:
            # Stay and monitor current zone
            pass

        elif action in [1, 2, 3, 4]:
            # Switch focus to a specific zone
            target_zone = action - 1
            if target_zone < self.n_zones:
                self.current_zone = target_zone

        elif action == 5:
            # Trigger alarm in current zone
            if self.intruder_active and self.intruder_zone == self.current_zone:
                # Correct alarm
                reward += 20.0
                self._clear_intruder()
            else:
                # False alarm
                reward -= 5.0

        elif action == 6:
            # Dispatch guard to current zone (if not already busy)
            if not self.guard_busy:
                self.guard_busy = True
                self.guard_zone = self.current_zone
                self.guard_eta = self.guard_travel_time
                reward -= 2.0  # cost of dispatch
            else:
                # Wasting time trying to dispatch when guard is busy
                reward -= 1.0

        # 2. Monitoring detection: if focusing on intruder zone, detect early
        if self.intruder_active and self.current_zone == self.intruder_zone:
            # Early detection without alarm
            reward += 15.0
            self._clear_intruder()

        # 3. Advance world dynamics
        self._update_guard(reward_ref=reward)
        self._update_intruder(reward_ref=reward, terminated_ref=terminated)

        # 4. Update features
        self._update_motion()
        self._update_time_since_scan()

        # 5. Check time limit
        if self.time_step >= self.max_steps:
            truncated = True

        observation = self._get_obs()

        # Info for analysis
        info.update(
            {
                "time_step": self.time_step,
                "intruder_active": self.intruder_active,
                "intruder_zone": self.intruder_zone,
                "intruder_progress": self.intruder_progress,
                "guard_busy": self.guard_busy,
                "guard_zone": self.guard_zone,
                "guard_eta": self.guard_eta,
            }
        )

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Normalize time_since_scan by a constant window
        max_scan_time = max(self.max_steps, 1)
        time_since_scan_norm = np.clip(
            self.time_since_scan / max_scan_time, 0.0, 1.0
        )

        # Zone-level features
        zone_features = np.concatenate(
            [
                self.motion,
                time_since_scan_norm,
                self.critical_zones,
            ],
            axis=0,
        )

        # Global features
        time_step_norm = self.time_step / float(self.max_steps)
        global_features = np.array(
            [
                float(self.guard_busy),
                float(time_step_norm),
                float(self.intruder_active),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([zone_features, global_features], axis=0)
        return obs.astype(np.float32)

    # ------------ Internal dynamics ------------ #

    def _spawn_intruder(self):
        # Spawn intruder in a random zone
        self.intruder_active = True
        self.intruder_zone = int(self.np_random.integers(0, self.n_zones))
        self.intruder_progress = 0.0

    def _clear_intruder(self):
        self.intruder_active = False
        self.intruder_zone = None
        self.intruder_progress = 0.0

    def _update_guard(self, reward_ref):
        # Move guard if dispatched
        if self.guard_busy:
            if self.guard_eta > 0:
                self.guard_eta -= 1

            if self.guard_eta <= 0:
                # Guard arrives at target zone
                if self.intruder_active and self.guard_zone == self.intruder_zone:
                    # Guard catches intruder
                    reward_ref += 20.0
                    self._clear_intruder()

                # Guard is now free
                self.guard_busy = False
                self.guard_zone = None
                self.guard_eta = 0

    def _update_intruder(self, reward_ref, terminated_ref):
        # Possibly spawn new intruder if none
        if not self.intruder_active:
            if self.np_random.random() < self.intruder_spawn_prob:
                self._spawn_intruder()
            return

        # Intruder moves closer to asset
        step_inc = 1.0 / max(self.max_intruder_steps, 1)
        self.intruder_progress += step_inc

        # If intruder reaches the asset and is still active -> catastrophic failure
        if self.intruder_progress >= 1.0 and self.intruder_active:
            # Large negative reward
            reward_ref += -30.0
            self._clear_intruder()
            # Mark episode as terminated by breach
            # (we can't directly set terminated_ref since it's a float,
            # so we will handle termination logic in step if needed.)
            # Here we simply store a flag; the calling step can check.
            self._breach_occurred = True
        else:
            self._breach_occurred = False

    def _update_motion(self):
        # Simple heuristic: intruder presence causes motion in that zone
        self.motion[:] = 0.0
        if self.intruder_active and self.intruder_zone is not None:
            self.motion[self.intruder_zone] = 1.0

    def _update_time_since_scan(self):
        # Increase time since last scan for all zones
        self.time_since_scan += 1.0
        # When focusing on a zone, reset its scan timer
        self.time_since_scan[self.current_zone] = 0.0

    # ------------ Rendering ------------ #

    def render(self, mode="ansi"):
        """
        Simple text-based rendering. For high-quality 2D visualization,
        use the pygame renderer in environment/rendering.py.
        """
        if mode == "ansi":
            lines = []
            lines.append(f"Step: {self.time_step}")
            lines.append(f"Current zone: {self.current_zone}")
            lines.append(f"Intruder active: {self.intruder_active}")
            lines.append(f"Intruder zone: {self.intruder_zone}")
            lines.append(f"Intruder progress: {self.intruder_progress:.2f}")
            lines.append(f"Guard busy: {self.guard_busy}")
            lines.append(f"Guard zone: {self.guard_zone}")
            lines.append(f"Guard ETA: {self.guard_eta}")
            lines.append(f"Motion: {self.motion}")
            lines.append(f"Time since scan: {self.time_since_scan}")
            return "\n".join(lines)
        else:
            # You can integrate with pygame here if you want,
            # but the main visualisation is handled in rendering.py.
            print(self.render(mode="ansi"))

    def close(self):
        pass