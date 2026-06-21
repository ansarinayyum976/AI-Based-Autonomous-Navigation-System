"""
Custom 2D Grid-World Navigation Environment
Simulates a mobile robot with LIDAR-like sensing navigating to a goal
while avoiding static and dynamic obstacles.

This replaces Gazebo/ROS with a lightweight pure-Python simulation
that runs instantly with no external robotics software required.
"""

import numpy as np
import math
import random


class RobotNavEnv:
    """
    A 2D continuous-space robot navigation environment.

    State (26-dim):
        - 24 simulated LIDAR ray distances (normalized 0-1)
        - Distance to goal (normalized 0-1)
        - Angle to goal (normalized -1 to 1)

    Actions (discrete, 5):
        0 = Forward
        1 = Slight Left Turn
        2 = Slight Right Turn
        3 = Sharp Left Turn
        4 = Sharp Right Turn
    """

    def __init__(self, world_size=10.0, num_static_obstacles=5,
                 num_dynamic_obstacles=0, max_steps=500):
        self.world_size = world_size
        self.num_static_obstacles = num_static_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.max_steps = max_steps

        self.num_lidar_rays = 24
        self.lidar_max_range = 3.5
        self.collision_threshold = 0.18
        self.goal_threshold = 0.35

        self.action_space_n = 5
        self.observation_space_shape = (26,)

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.robot_x = random.uniform(1.0, 2.0)
        self.robot_y = random.uniform(1.0, 2.0)
        self.robot_theta = random.uniform(0, 2 * math.pi)

        self.goal_x = random.uniform(self.world_size - 2.0, self.world_size - 1.0)
        self.goal_y = random.uniform(self.world_size - 2.0, self.world_size - 1.0)

        # Static obstacles (circles) - spaced apart from each other too
        self.static_obstacles = []
        for _ in range(self.num_static_obstacles):
            attempts = 0
            while attempts < 30:
                ox = random.uniform(1.8, self.world_size - 1.8)
                oy = random.uniform(1.8, self.world_size - 1.8)
                ok = (math.dist((ox, oy), (self.robot_x, self.robot_y)) > 1.5 and
                      math.dist((ox, oy), (self.goal_x, self.goal_y)) > 1.5)
                # Keep obstacles from clustering too tightly together
                for existing in self.static_obstacles:
                    if math.dist((ox, oy), (existing['x'], existing['y'])) < 1.3:
                        ok = False
                        break
                if ok:
                    self.static_obstacles.append({'x': ox, 'y': oy, 'r': random.uniform(0.25, 0.40)})
                    break
                attempts += 1

        # Dynamic obstacles (move in circular patrol patterns)
        self.dynamic_obstacles = []
        for _ in range(self.num_dynamic_obstacles):
            cx = random.uniform(2, self.world_size - 2)
            cy = random.uniform(2, self.world_size - 2)
            self.dynamic_obstacles.append({
                'cx': cx, 'cy': cy, 'radius': random.uniform(0.5, 1.0),
                'angle': random.uniform(0, 2 * math.pi),
                'speed': random.uniform(0.015, 0.03), 'r': 0.22
            })
            self._update_dynamic_obstacle_pos(self.dynamic_obstacles[-1])

        self.prev_dist_to_goal = self._dist_to_goal()
        self.trajectory = [(self.robot_x, self.robot_y)]

        return self._get_state()

    def _update_dynamic_obstacle_pos(self, obs):
        obs['angle'] += obs['speed']
        obs['x'] = obs['cx'] + obs['radius'] * math.cos(obs['angle'])
        obs['y'] = obs['cy'] + obs['radius'] * math.sin(obs['angle'])

    def _all_obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def _dist_to_goal(self):
        return math.dist((self.robot_x, self.robot_y), (self.goal_x, self.goal_y))

    def _angle_to_goal(self):
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.robot_theta
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return angle_diff

    def _cast_lidar(self):
        """Simulate 24 LIDAR rays around the robot, returning normalized distances."""
        readings = []
        for i in range(self.num_lidar_rays):
            ray_angle = self.robot_theta + (2 * math.pi * i / self.num_lidar_rays)
            min_dist = self.lidar_max_range

            # Check against world boundaries
            for t in np.linspace(0.05, self.lidar_max_range, 40):
                rx = self.robot_x + t * math.cos(ray_angle)
                ry = self.robot_y + t * math.sin(ray_angle)
                if rx < 0 or rx > self.world_size or ry < 0 or ry > self.world_size:
                    min_dist = min(min_dist, t)
                    break

            # Check against obstacles
            for obs in self._all_obstacles():
                ox, oy, orad = obs.get('x', obs.get('cx')), obs.get('y', obs.get('cy')), obs['r']
                d = self._ray_circle_intersect(self.robot_x, self.robot_y, ray_angle, ox, oy, orad)
                if d is not None:
                    min_dist = min(min_dist, d)

            readings.append(min(min_dist, self.lidar_max_range) / self.lidar_max_range)
        return readings

    @staticmethod
    def _ray_circle_intersect(rx, ry, angle, cx, cy, radius):
        dx, dy = math.cos(angle), math.sin(angle)
        fx, fy = rx - cx, ry - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        disc_sqrt = math.sqrt(disc)
        t1 = (-b - disc_sqrt) / (2 * a)
        t2 = (-b + disc_sqrt) / (2 * a)
        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None

    def _get_state(self):
        lidar = self._cast_lidar()
        dist_norm = min(self._dist_to_goal() / (self.world_size * 1.414), 1.0)
        angle_norm = self._angle_to_goal() / math.pi
        return np.array(lidar + [dist_norm, angle_norm], dtype=np.float32)

    def _check_collision(self):
        for obs in self._all_obstacles():
            ox, oy = obs.get('x', obs.get('cx')), obs.get('y', obs.get('cy'))
            if math.dist((self.robot_x, self.robot_y), (ox, oy)) < (obs['r'] + self.collision_threshold):
                return True
        margin = self.collision_threshold
        if (self.robot_x < margin or self.robot_x > self.world_size - margin or
                self.robot_y < margin or self.robot_y > self.world_size - margin):
            return True
        return False

    def step(self, action):
        self.steps += 1

        # Action -> (linear_v, angular_v)
        action_map = {
            0: (0.15, 0.0),
            1: (0.10, 0.26),
            2: (0.10, -0.26),
            3: (0.05, 0.52),
            4: (0.05, -0.52),
        }
        linear_v, angular_v = action_map.get(action, (0.0, 0.0))

        self.robot_theta += angular_v
        self.robot_x += linear_v * math.cos(self.robot_theta)
        self.robot_y += linear_v * math.sin(self.robot_theta)

        for obs in self.dynamic_obstacles:
            self._update_dynamic_obstacle_pos(obs)

        self.trajectory.append((self.robot_x, self.robot_y))

        curr_dist = self._dist_to_goal()
        goal_reached = curr_dist < self.goal_threshold
        collision = self._check_collision()
        timeout = self.steps >= self.max_steps

        # Reward function
        if goal_reached:
            reward = 100.0
        elif collision:
            reward = -100.0
        else:
            shaping = (self.prev_dist_to_goal - curr_dist) * 10
            reward = shaping - 0.1

        self.prev_dist_to_goal = curr_dist
        done = goal_reached or collision or timeout

        info = {'goal_reached': goal_reached, 'collision': collision, 'timeout': timeout}
        return self._get_state(), reward, done, info

    def render_state(self):
        """Return a dict representation for visualization/JSON."""
        return {
            'robot': {'x': self.robot_x, 'y': self.robot_y, 'theta': self.robot_theta},
            'goal': {'x': self.goal_x, 'y': self.goal_y},
            'static_obstacles': self.static_obstacles,
            'dynamic_obstacles': self.dynamic_obstacles,
            'trajectory': self.trajectory[-200:],
            'world_size': self.world_size,
            'steps': self.steps,
        }
