# test_ppac.py
from chiplet_gym.models.ppac_calculator import PPACCalculator
from chiplet_gym.rewards.ppac_reward import PPACReward
from chiplet_gym.utils.config import DEFAULT_CONFIG

calc = PPACCalculator(DEFAULT_CONFIG)

# Test a simple design
design = {
    "num_chiplets": 16,
    "placement_rows": 4,
    "placement_cols": 4,
    "num_hbm": 2,
    "hbm_locations": ["left", "right"],
    "interconnect_2_5d": "CoWoS",
    "data_rate_2_5d": 10,
    "link_count_2_5d": 500,
    "interconnect_3d": "SoIC",
    "data_rate_3d": 30,
    "link_count_3d": 1000,
}

ppac = calc.compute(design)

print("=== PPAC Values ===")
for key, val in ppac.items():
    if isinstance(val, float):
        print(
            f"{key:20s}: {val:.4e}"
            if val < 0.01 or val > 1000
            else f"{key:20s}: {val:.2f}"
        )
    else:
        print(f"{key:20s}: {val}")

print("\n=== Expected Ranges ===")
print("Throughput: 50-250 tasks/sec")
print("Energy: 1e-7 to 1e-5 J/task")
print("Cost: $100-$2000")


reward_fn = PPACReward(alpha=0.5, beta=0.3, gamma=0.2)
reward = reward_fn.compute(ppac)

print("\n=== Reward Calculation ===")
print(f"Throughput norm: {ppac['throughput'] / 100.0:.4f}")
print(f"Energy norm: {ppac['energy'] / 1e-4:.4f}")
print(f"Cost norm: {ppac['total_cost'] / 1000.0:.4f}")
print(f"Final reward: {reward:.4f}")
print("\nExpected reward range: -5 to +5")
