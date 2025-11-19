# environment/rendering.py

import sys
import random
import pygame
import numpy as np

from .custom_env import SmartBuildingSecurityEnv


class SecurityEnvRenderer:
    """
    Pygame-based visualization for SmartBuildingSecurityEnv.
    """

    def __init__(self, env: SmartBuildingSecurityEnv, width=800, height=600):
        self.env = env
        self.width = width
        self.height = height

        pygame.init()
        pygame.display.set_caption("Smart Building Security - RL Environment")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 16)

        # Layout for zones (simple 2x2 grid)
        self.zone_rects = self._compute_zone_rects()

    def _compute_zone_rects(self):
        cols = 2
        rows = int(np.ceil(self.env.n_zones / cols))
        margin = 20
        zone_w = (self.width - (cols + 1) * margin) // cols
        zone_h = (self.height - 200 - (rows + 1) * margin) // rows  # leave space for info text

        rects = []
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= self.env.n_zones:
                    break
                x = margin + c * (zone_w + margin)
                y = margin + r * (zone_h + margin)
                rects.append(pygame.Rect(x, y, zone_w, zone_h))
                idx += 1
        return rects

    def draw(self, info=None):
        self.screen.fill((18, 18, 30))

        # Colors
        zone_color = (40, 40, 70)
        focus_color = (80, 160, 255)
        critical_color = (150, 50, 50)
        intruder_color = (255, 80, 80)
        guard_color = (80, 200, 120)
        text_color = (220, 220, 220)

        # Draw zones
        for i, rect in enumerate(self.zone_rects):
            is_critical = self.env.critical_zones[i] > 0.5
            color = zone_color
            if is_critical:
                color = critical_color

            pygame.draw.rect(self.screen, color, rect, border_radius=8)

            # Highlight current focus zone
            if i == self.env.current_zone:
                pygame.draw.rect(self.screen, focus_color, rect, width=3, border_radius=8)

            # Motion indicator
            if self.env.motion[i] > 0.5:
                pygame.draw.circle(
                    self.screen,
                    (255, 200, 0),
                    (rect.centerx, rect.centery),
                    10,
                )

            # Zone label
            label = self.font.render(f"Zone {i}", True, text_color)
            self.screen.blit(label, (rect.x + 8, rect.y + 8))

        # Draw intruder
        if self.env.intruder_active and self.env.intruder_zone is not None:
            z_idx = self.env.intruder_zone
            if 0 <= z_idx < len(self.zone_rects):
                rect = self.zone_rects[z_idx]
                # Move intruder vertically with progress
                x = rect.centerx
                y = rect.bottom - int(rect.height * self.env.intruder_progress)
                pygame.draw.circle(self.screen, intruder_color, (x, y), 12)

        # Draw guard
        if self.env.guard_busy and self.env.guard_zone is not None:
            z_idx = self.env.guard_zone
            if 0 <= z_idx < len(self.zone_rects):
                rect = self.zone_rects[z_idx]
                x = rect.centerx
                y = rect.centery
                pygame.draw.circle(self.screen, guard_color, (x, y), 10)

        # Info panel at bottom
        panel_y = self.height - 180
        pygame.draw.rect(
            self.screen,
            (25, 25, 40),
            pygame.Rect(0, panel_y, self.width, 180),
        )

        lines = [
            f"Step: {self.env.time_step}",
            f"Current zone: {self.env.current_zone}",
            f"Intruder active: {self.env.intruder_active}",
            f"Intruder zone: {self.env.intruder_zone}",
            f"Intruder progress: {self.env.intruder_progress:.2f}",
            f"Guard busy: {self.env.guard_busy}",
            f"Guard zone: {self.env.guard_zone}",
            f"Guard ETA: {self.env.guard_eta}",
            f"Last action: {self.env.last_action}",
        ]

        if info is not None:
            lines.append(f"Reward (this step): {info.get('reward', 0.0):.2f}")

        y = panel_y + 10
        for line in lines:
            label = self.font.render(line, True, text_color)
            self.screen.blit(label, (10, y))
            y += 20

        pygame.display.flip()

    def close(self):
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                pygame.quit()
                sys.exit()


def run_random_agent(num_episodes=3, max_steps_per_episode=100, fps=8):
    """
    Run a random policy in the environment and visualize it with pygame.

    Use a screen recorder to capture this as your 'static file' of random actions.
    """
    env = SmartBuildingSecurityEnv()
    renderer = SecurityEnvRenderer(env)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        t = 0

        while not done and not truncated and t < max_steps_per_episode:
            renderer.handle_events()

            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            info["reward"] = reward

            renderer.draw(info=info)
            renderer.clock.tick(fps)

            done = terminated
            obs = next_obs
            t += 1

    renderer.close()
    env.close()


if __name__ == "__main__":
    run_random_agent()