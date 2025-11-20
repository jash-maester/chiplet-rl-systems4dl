"""
PPAC-based Reward Function
Multi-objective reward: maximize throughput, minimize energy and cost
"""

from typing import Dict

import numpy as np


class PPACReward:
    """
    Weighted PPAC reward function
    reward = alpha * throughput - beta * energy - gamma * cost

    Default weights (configurable):
        alpha = 0.5 (throughput weight)
        beta = 0.3 (energy weight)
        gamma = 0.2 (cost weight)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Normalization constants (based on typical ranges)
        self.throughput_norm = 100.0  # tasks/sec
        # self.throughput_norm = 200.0  # tasks/sec
        self.energy_norm = 1e-4  # joules per task
        # self.energy_norm = 1e-6  # joules per task
        self.cost_norm = 1000.0  # USD
        # self.cost_norm = 500.0  # USD

    def compute(self, ppac: Dict) -> float:
        """
        Compute reward from PPAC metrics

        Args:
            ppac: dict with keys 'throughput', 'energy', 'total_cost'

        Returns:
            reward: scalar value
        """
        # Clip values to prevent explosions
        throughput = np.clip(ppac["throughput"], 0, 500)  # Max 500 tasks/sec
        energy = np.clip(ppac["energy"], 1e-8, 1e-4)  # Reasonable energy range
        cost = np.clip(ppac["total_cost"], 10, 10000)  # $10 - $10k range

        # Normalize
        throughput_norm = throughput / self.throughput_norm
        energy_norm = energy / self.energy_norm
        cost_norm = cost / self.cost_norm

        # Normalize metrics
        # throughput_norm = ppac["throughput"] / self.throughput_norm
        # energy_norm = ppac["energy"] / self.energy_norm
        # cost_norm = ppac["total_cost"] / self.cost_norm

        # Weighted sum (maximize throughput, minimize energy and cost)
        reward = (
            self.alpha * throughput_norm
            - self.beta * energy_norm
            - self.gamma * cost_norm
        )
        # Clip final reward to prevent value explosion
        reward = np.clip(reward, -10, 10)

        return reward
