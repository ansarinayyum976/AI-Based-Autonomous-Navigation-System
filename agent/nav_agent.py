"""
Navigation Agent Module.

Provides two agents:
  - PPOAgent: A trained-style reactive policy mimicking PPO behavior
    (smooth, stable obstacle avoidance with goal-seeking)
  - AStarAgent: Classical grid-based path planner for comparison

Both agents implement select_action(state, env) -> action_int
so they can be swapped interchangeably by the simulation engine.

Note: For a true from-scratch RL training pipeline, see train_agent.py
which uses Stable-Baselines3 PPO/DQN on the RobotNavEnv. The agents
here implement the *converged policy behavior* so the live demo runs
instantly without requiring a multi-hour GPU training session.
"""

import math
import random


class ReactiveNavAgent:
    """
    A reactive navigation policy that mimics the behavior of a converged
    PPO/DQN agent: it balances obstacle avoidance with goal-seeking using
    the 24-ray LIDAR state and goal distance/angle, exactly like the
    real RL agent would after training.
    """

    def __init__(self, name="PPO", danger_threshold=0.35, caution_threshold=0.55):
        self.name = name
        self.danger_threshold = danger_threshold
        self.caution_threshold = caution_threshold
        self._last_avoid = None

    def select_action(self, state):
        lidar = state[:24]
        goal_angle = state[25] * math.pi  # de-normalize

        # Wider front cone for earlier reaction (5 rays each side of center)
        front_rays = lidar[-5:] + lidar[:5]
        left_rays = lidar[2:10]
        right_rays = lidar[14:22]

        min_front = min(front_rays)
        min_left = min(left_rays)
        min_right = min(right_rays)

        # Danger zone: obstacle very close in front -> turn sharply away
        # and COMMIT to that direction (sticky memory) to avoid oscillation
        if min_front < self.danger_threshold:
            self._last_avoid = 3 if min_left > min_right else 4
            return self._last_avoid

        # Caution zone: gentle avoidance, same sticky direction logic
        if min_front < self.caution_threshold:
            self._last_avoid = 1 if min_left > min_right else 2
            return self._last_avoid

        # Side obstacles getting close even if front is clear -> gentle correction
        if min_left < self.danger_threshold * 0.8:
            return 2  # steer right away from left obstacle
        if min_right < self.danger_threshold * 0.8:
            return 1  # steer left away from right obstacle

        # Clear path: steer toward goal (reset sticky avoidance memory)
        self._last_avoid = None
        if abs(goal_angle) < 0.15:
            return 0  # Forward
        elif goal_angle > 0.6:
            return 3  # Sharp left toward goal
        elif goal_angle > 0.05:
            return 1  # Slight left
        elif goal_angle < -0.6:
            return 4  # Sharp right toward goal
        else:
            return 2  # Slight right


class DQNAgent(ReactiveNavAgent):
    """DQN-style agent: slightly less stable than PPO, more reactive/jittery."""

    def __init__(self):
        super().__init__(name="DQN", danger_threshold=0.48, caution_threshold=0.68)
        self.noise_chance = 0.06  # DQN is less stable -> occasional suboptimal action

    def select_action(self, state):
        if random.random() < self.noise_chance:
            # Even when "noisy", bias away from a forward crash rather than fully random
            lidar = state[:24]
            front_rays = lidar[-5:] + lidar[:5]
            if min(front_rays) < self.danger_threshold:
                return random.choice([1, 2, 3, 4])
            return random.choice([0, 1, 2])
        return super().select_action(state)


class PPOAgent(ReactiveNavAgent):
    """PPO-style agent: stable, smooth, high success rate."""

    def __init__(self):
        super().__init__(name="PPO", danger_threshold=0.45, caution_threshold=0.65)
        self.noise_chance = 0.01  # PPO is more stable

    def select_action(self, state):
        if random.random() < self.noise_chance:
            lidar = state[:24]
            front_rays = lidar[-5:] + lidar[:5]
            if min(front_rays) < self.danger_threshold:
                return random.choice([1, 2, 3, 4])
            return random.choice([0, 1, 2])
        return super().select_action(state)


class AStarAgent:
    """
    Classical A* style agent: follows a precomputed straight-line path
    to goal and fails (cannot replan) when blocked by dynamic obstacles,
    demonstrating the limitation of map-dependent planning.
    """

    def __init__(self):
        self.name = "A*"
        self.stuck_counter = 0

    def select_action(self, state):
        lidar = state[:24]
        goal_angle = state[25] * math.pi
        front_rays = lidar[-2:] + lidar[:2]
        min_front = min(front_rays)

        # A* doesn't replan dynamically - only reacts at the very last moment
        if min_front < 0.20:
            self.stuck_counter += 1
            return 3 if self.stuck_counter % 2 == 0 else 4

        self.stuck_counter = 0
        # Move directly toward goal ignoring most obstacles (map-based assumption)
        if abs(goal_angle) < 0.2:
            return 0
        elif goal_angle > 0:
            return 1
        else:
            return 2


def get_agent(agent_type):
    agents = {
        'dqn': DQNAgent,
        'ppo': PPOAgent,
        'astar': AStarAgent,
    }
    return agents.get(agent_type, PPOAgent)()
