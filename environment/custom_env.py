# environment/custom_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# NEW: pygame for visualization
import pygame


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

        # Track breaches (used internally)
        self._breach_occurred = False

        # ---- Pygame rendering state (for mode="human") ----
        self.screen = None
        self.clock = None
        self.font = None
        self.window_width = 800
        self.window_height = 400
        self.zone_width = 120
        self.zone_height = 150
        self.zone_margin = 20
        self.top_margin = 80

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
        self._breach_occurred = False

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
        # NOTE: reward_ref is a float, so changes here won't propagate back.
        # We keep your original logic to not break existing training.
        if self.guard_busy:
            if self.guard_eta > 0:
                self.guard_eta -= 1

            if self.guard_eta <= 0:
                # Guard arrives at target zone
                if self.intruder_active and self.guard_zone == self.intruder_zone:
                    # Guard catches intruder
                    # reward_ref += 20.0  # logically should add reward
                    self._clear_intruder()

                # Guard is now free
                self.guard_busy = False
                self.guard_zone = None
                self.guard_eta = 0

    def _update_intruder(self, reward_ref, terminated_ref):
        # NOTE: same as above: reward_ref, terminated_ref not updated outside.
        if not self.intruder_active:
            if self.np_random.random() < self.intruder_spawn_prob:
                self._spawn_intruder()
            return

        # Intruder moves closer to asset
        step_inc = 1.0 / max(self.max_intruder_steps, 1)
        self.intruder_progress += step_inc

        # If intruder reaches the asset and is still active -> catastrophic failure
        if self.intruder_progress >= 1.0 and self.intruder_active:
            # reward_ref += -30.0  # logically should add penalty
            self._clear_intruder()
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
        mode="ansi": return a text representation.
        mode="human": open a pygame window and visualize zones, agent, intruder, guard.
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

        if mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.window_width, self.window_height)
                )
                pygame.display.set_caption("Smart Building Security - RL Agent")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("arial", 18)

            # Handle window close events so it doesn't freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.screen = None
                    return

            # Background
            self.screen.fill((20, 20, 30))

            # Draw zones as rectangles
            start_x = self.zone_margin
            y = self.top_margin

            for z in range(self.n_zones):
                x = start_x + z * (self.zone_width + self.zone_margin)

                # Color based on criticality
                if self.critical_zones[z] == 1.0:
                    base_color = (180, 60, 60)  # reddish for critical
                else:
                    base_color = (60, 120, 180)  # bluish for normal

                rect = pygame.Rect(x, y, self.zone_width, self.zone_height)

                # Highlight current focus
                if z == self.current_zone:
                    pygame.draw.rect(self.screen, (250, 250, 0), rect, border_radius=8)
                    inner_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, base_color, inner_rect, border_radius=8)
                else:
                    pygame.draw.rect(self.screen, base_color, rect, border_radius=8)

                # Text: zone label
                label = self.font.render(f"Zone {z}", True, (255, 255, 255))
                self.screen.blit(label, (x + 10, y + 10))

                # Motion indicator
                motion_val = self.motion[z]
                if motion_val > 0.5:
                    m_text = self.font.render("MOTION", True, (255, 200, 0))
                else:
                    m_text = self.font.render("No motion", True, (180, 180, 180))
                self.screen.blit(m_text, (x + 10, y + 40))

                # Time since scan
                t_text = self.font.render(
                    f"Last scan: {int(self.time_since_scan[z])}", True, (200, 200, 200)
                )
                self.screen.blit(t_text, (x + 10, y + 65))

                # Intruder
                if self.intruder_active and self.intruder_zone == z:
                    intr_x = x + self.zone_width // 2
                    intr_y = y + self.zone_height - 30
                    pygame.draw.circle(self.screen, (255, 50, 50), (intr_x, intr_y), 12)

                # Guard
                if self.guard_busy and self.guard_zone == z:
                    guard_x = x + self.zone_width // 2
                    guard_y = y + self.zone_height // 2
                    pygame.draw.circle(self.screen, (50, 255, 50), (guard_x, guard_y), 10)

            # Status text at the top
            status_y = 10
            info_lines = [
                f"Step: {self.time_step}/{self.max_steps}",
                f"Current zone: {self.current_zone}",
                f"Intruder active: {self.intruder_active} (zone={self.intruder_zone})",
                f"Intruder progress: {self.intruder_progress:.2f}",
                f"Guard busy: {self.guard_busy} (zone={self.guard_zone}, eta={self.guard_eta})",
                f"Last action: {self.last_action}",
            ]
            for i, text in enumerate(info_lines):
                surf = self.font.render(text, True, (230, 230, 230))
                self.screen.blit(surf, (20, status_y + i * 20))

            pygame.display.flip()
            # Limit FPS
            self.clock.tick(self.metadata.get("render_fps", 10))
        else:
            # Fallback to ANSI if unknown mode
            print(self.render(mode="ansi"))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None