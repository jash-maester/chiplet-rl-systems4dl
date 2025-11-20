"""
Quick test script to verify environment works
Run this before training to catch issues early
"""

import chiplet_gym  # noqa: F401
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env


def test_environment():
    """Test basic environment functionality"""
    print("Creating environment...")
    env = gym.make("ChipletEnv-v0")

    print("\nRunning check_env()...")
    check_env(env, warn=True)

    print("\nTesting random agent for 10 steps...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: reward={reward:.2f}, valid={info.get('valid', False)}")

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("\n✅ Environment test passed!")


if __name__ == "__main__":
    test_environment()
