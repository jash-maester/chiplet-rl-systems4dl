"""
Chiplet-Gym: RL-based Optimization for Chiplet AI Accelerators
Compatible with Stable-Baselines3
"""

from gymnasium.envs.registration import register

register(
    id="ChipletEnv-v0",
    entry_point="chiplet_gym.envs:ChipletEnv",
    max_episode_steps=50,
)

__version__ = "0.1.0"
