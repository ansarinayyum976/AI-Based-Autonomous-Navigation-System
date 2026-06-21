"""
train_agent.py

REAL reinforcement learning training script using Stable-Baselines3.
This trains actual DQN and PPO agents on the RobotNavEnv from scratch.

Usage:
    python train_agent.py --algo ppo --timesteps 500000
    python train_agent.py --algo dqn --timesteps 500000

Requirements:
    pip install stable-baselines3 gymnasium

Note: This is the FULL training pipeline referenced in the project report.
Training takes 30-90 minutes on CPU, faster with GPU. The web demo
(app.py) uses a lightweight reactive policy (agent/nav_agent.py) so
it can run instantly without requiring this training step first.
Run this script if you want to train and export your own real
Stable-Baselines3 model checkpoint.
"""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.grid_world import RobotNavEnv


class GymRobotNavEnv(gym.Env):
    """Gymnasium-compatible wrapper around RobotNavEnv for Stable-Baselines3."""

    metadata = {'render_modes': []}

    def __init__(self, num_dynamic_obstacles=0):
        super().__init__()
        self.env = RobotNavEnv(num_dynamic_obstacles=num_dynamic_obstacles)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(26,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        state = self.env.reset(seed=seed)
        return state, {}

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        terminated = info['goal_reached'] or info['collision']
        truncated = info['timeout']
        return state, reward, terminated, truncated, info


def train(algo='ppo', timesteps=500000, num_dynamic_obstacles=4, save_path=None):
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    def make_env():
        return Monitor(GymRobotNavEnv(num_dynamic_obstacles=num_dynamic_obstacles))

    env = make_vec_env(make_env, n_envs=1)

    if algo == 'ppo':
        model = PPO(
            'MlpPolicy', env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, clip_range=0.2,
            ent_coef=0.01, verbose=1,
            tensorboard_log='./tb_logs/ppo'
        )
    elif algo == 'dqn':
        model = DQN(
            'MlpPolicy', env,
            learning_rate=5e-4, buffer_size=100000, batch_size=128,
            gamma=0.99, exploration_fraction=0.4,
            exploration_final_eps=0.01, target_update_interval=500,
            verbose=1,
            tensorboard_log='./tb_logs/dqn'
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"\nTraining {algo.upper()} for {timesteps} timesteps...\n")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    out_path = save_path or f'models/{algo}_navigation_model'
    model.save(out_path)
    print(f"\nModel saved to {out_path}.zip")
    return model


def evaluate(model, num_episodes=100, num_dynamic_obstacles=4):
    env = GymRobotNavEnv(num_dynamic_obstacles=num_dynamic_obstacles)
    successes, collisions, steps_list = 0, 0, []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
        if info.get('goal_reached'):
            successes += 1
            steps_list.append(steps)
        if info.get('collision'):
            collisions += 1

    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"  Success Rate:   {successes/num_episodes*100:.1f}%")
    print(f"  Collision Rate: {collisions/num_episodes*100:.1f}%")
    if steps_list:
        print(f"  Avg Steps (success): {np.mean(steps_list):.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['ppo', 'dqn'], default='ppo')
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--dynamic-obstacles', type=int, default=4)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    model = train(algo=args.algo, timesteps=args.timesteps,
                  num_dynamic_obstacles=args.dynamic_obstacles)
    evaluate(model, num_episodes=100, num_dynamic_obstacles=args.dynamic_obstacles)
