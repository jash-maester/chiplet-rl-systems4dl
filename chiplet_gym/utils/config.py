"""
Default configuration for Chiplet-Gym
"""

DEFAULT_CONFIG = {
    # Environment parameters
    "max_steps": 50,  # Maximum steps per episode
    "max_package_area": 1000.0,  # mm^2
    # Reward weights
    "reward_alpha": 0.5,  # Throughput weight
    "reward_beta": 0.3,  # Energy weight
    "reward_gamma": 0.2,  # Cost weight
    # Design constraints
    "min_chiplets": 4,
    "max_chiplets": 128,
    "min_hbm": 1,
    "max_hbm": 5,
    # Workload (generic AI)
    "workload": "generic",
    # Random seed
    "seed": 42,
}
