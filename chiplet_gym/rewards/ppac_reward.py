"""
PPAC-based Reward Function
Multi-objective reward: maximize throughput, minimize energy and cost
"""

from typing import Dict


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
        self.throughput_norm = 200.0  # tasks/sec
        self.energy_norm = 1e-6  # joules per task
        self.cost_norm = 500.0  # USD

    def compute(self, ppac: Dict) -> float:
        """
        Compute reward from PPAC metrics

        Args:
            ppac: dict with keys 'throughput', 'energy', 'total_cost'

        Returns:
            reward: scalar value
        """
        # Normalize metrics
        throughput_norm = ppac["throughput"] / self.throughput_norm
        energy_norm = ppac["energy"] / self.energy_norm
        cost_norm = ppac["total_cost"] / self.cost_norm

        # Weighted sum (maximize throughput, minimize energy and cost)
        reward = (
            self.alpha * throughput_norm
            - self.beta * energy_norm
            - self.gamma * cost_norm
        )

        return reward
