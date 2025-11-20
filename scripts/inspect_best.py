import chiplet_gym
import gymnasium as gym
from stable_baselines3 import PPO

# Load best model
model = PPO.load("./models/best/best_model.zip")
env = gym.make("ChipletEnv-v0")

# Run one episode deterministically
obs, info = env.reset(seed=42)
total_reward = 0

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step == 0 and info.get("valid", False):
        print("=== Agent's Learned Design ===")
        print(f"Chiplets: {info['design_params']['num_chiplets']}")
        print(
            f"Placement: {info['design_params']['placement_rows']}×{info['design_params']['placement_cols']}"
        )
        print(
            f"HBM: {info['design_params']['num_hbm']} at {info['design_params']['hbm_locations']}"
        )
        print(
            f"2.5D: {info['design_params']['interconnect_2_5d']} @ {info['design_params']['data_rate_2_5d']} Gbps"
        )
        print(
            f"3D: {info['design_params']['interconnect_3d']} @ {info['design_params']['data_rate_3d']} Gbps"
        )
        print("\n=== PPAC Metrics ===")
        print(f"Throughput: {info['throughput']:.2f} tasks/sec")
        print(f"Energy: {info['energy']:.2e} J/task")
        print(f"Cost: ${info['cost']:.2f}")
        print(f"Reward: {reward:.2f}")

    if terminated or truncated:
        break

env.close()
print(f"\nTotal Episode Reward: {total_reward:.2f}")
