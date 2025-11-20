"""
Evaluation script for trained Chiplet-Gym agents
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from chiplet_gym.utils.config import DEFAULT_CONFIG


def evaluate_agent(model_path, num_episodes=10):
    """Evaluate trained agent"""

    # Load environment
    env = gym.make("ChipletEnv-v0", config=DEFAULT_CONFIG)

    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Evaluation loop
    episode_rewards = []
    episode_throughputs = []
    episode_energies = []
    episode_costs = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        if info.get("valid", False):
            episode_rewards.append(episode_reward)
            episode_throughputs.append(info["throughput"])
            episode_energies.append(info["energy"])
            episode_costs.append(info["cost"])

            print(f"\nEpisode {episode + 1}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Throughput: {info['throughput']:.2f} tasks/sec")
            print(f"  Energy: {info['energy']:.2e} J/task")
            print(f"  Cost: ${info['cost']:.2f}")
            print(f"  Design: {info['ppac']}")

    # Summary statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(
        f"Avg Throughput: {np.mean(episode_throughputs):.2f} ± {np.std(episode_throughputs):.2f}"
    )
    print(
        f"Avg Energy: {np.mean(episode_energies):.2e} ± {np.std(episode_energies):.2e}"
    )
    print(f"Avg Cost: ${np.mean(episode_costs):.2f} ± ${np.std(episode_costs):.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    evaluate_agent(args.model_path, args.num_episodes)
