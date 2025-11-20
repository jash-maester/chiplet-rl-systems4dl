"""
Main Chiplet-Gym Environment - SB3 Compatible
Minimal observation space, configurable action space
"""

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chiplet_gym.models.ppac_calculator import PPACCalculator
from chiplet_gym.rewards.ppac_reward import PPACReward
from chiplet_gym.utils.config import DEFAULT_CONFIG


class ChipletEnv(gym.Env):
    """
    Chiplet-based AI Accelerator Design Environment

    Action Space (MultiDiscrete):
        - num_chiplets: [4, 128] - number of AI chiplets
        - placement_rows: [1, 16] - chiplet array rows (for mesh topology)
        - num_hbm: [1, 5] - number of HBM chiplets
        - hbm_location: [0, 31] - 5-bit encoding for 5 possible locations
        - interconnect_2_5d: [0, 1] - CoWoS=0, EMIB=1
        - data_rate_2_5d: [1, 20] - Gbps
        - link_count_2_5d_idx: [0, 99] - maps to [50, 5000] step 50
        - interconnect_3d: [0, 1] - SoIC=0, FOVEROS=1
        - data_rate_3d: [20, 50] - Gbps
        - link_count_3d_idx: [0, 99] - maps to [100, 10000] step 100

    Observation Space (Box):
        - Normalized state vector (32-dim for minimal config)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # Load configuration
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        # Action space: 10 design parameters (simplified from Table 1)
        self.action_space = spaces.MultiDiscrete(
            [
                125,  # num_chiplets: 4-128 (index 0-124 maps to 4-128)
                16,  # placement_rows: 1-16
                5,  # num_hbm: 1-5
                32,  # hbm_location: 5-bit bitmask for [left, right, top, bottom, middle]
                2,  # interconnect_2.5D: CoWoS=0, EMIB=1
                20,  # data_rate_2.5D: 1-20 Gbps
                100,  # link_count_2.5D: index maps to 50-5000 step 50
                2,  # interconnect_3D: SoIC=0, FOVEROS=1
                31,  # data_rate_3D: 20-50 Gbps
                100,  # link_count_3D: index maps to 100-10000 step 100
            ]
        )

        # Observation space: 32-dim minimal state
        # [area_budget, current_area_per_chiplet, latency_ai2ai, latency_hbm2ai,
        #  energy, cost, throughput, utilization, ... (24 more for extensibility)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(32,), dtype=np.float32
        )

        # Initialize PPAC calculator and reward function
        self.ppac_calc = PPACCalculator(self.config)
        self.reward_fn = PPACReward(
            alpha=self.config["reward_alpha"],
            beta=self.config["reward_beta"],
            gamma=self.config["reward_gamma"],
        )

        # State tracking
        self.current_step = 0
        self.state = None
        self.best_reward = -np.inf

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize random design
        self.current_step = 0
        self.state = self._init_state()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        design_params = self._decode_action(action)

        # Always compute PPAC (even for invalid designs)
        try:
            ppac = self.ppac_calc.compute(design_params)
            valid = True
        except Exception:
            # If PPAC computation fails, create dummy metrics
            ppac = {
                "throughput": 0.0,
                "energy": 1e-3,
                "total_cost": 10000.0,
                "area": design_params["num_chiplets"] * 20,
            }
            valid = False

        # Shaped reward (not binary valid/invalid)
        if valid and self._is_valid_design(design_params):
            # Normal PPAC reward
            reward = self.reward_fn.compute(ppac)
        else:
            # Partial credit based on how "close" to valid
            # Guide agent towards smaller chiplet counts and valid placements
            penalty = 0
            if design_params["num_chiplets"] > 64:
                penalty += (design_params["num_chiplets"] - 64) * 0.1
            if design_params["num_hbm"] > len(design_params["hbm_locations"]):
                penalty += 5

            reward = -penalty - 1  # Small negative, not -1000

        terminated = self.current_step >= self.config["max_steps"]
        truncated = False

        # Update state
        # self.state["throughput"] = ppac.get("throughput", 0)
        # self.state["energy"] = ppac.get("energy", 1e-3)
        # self.state["cost"] = ppac.get("total_cost", 10000)

        # Update state with bounded values
        self.state["area_per_chiplet"] = ppac["chiplet_area"]
        self.state["latency_ai2ai"] = ppac["latency_ai2ai"]
        self.state["latency_hbm2ai"] = ppac["latency_hbm2ai"]
        self.state["energy"] = min(ppac["energy"], 1e-4)  # Clip
        self.state["cost"] = min(ppac["total_cost"], 10000)  # Clip
        self.state["throughput"] = min(ppac["throughput"], 500)  # Clip

        info = {
            "valid": valid,
            "throughput": ppac.get("throughput", 0),
            "energy": ppac.get("energy", 1e-3),
            "cost": ppac.get("total_cost", 10000),
            "ppac": ppac,
            "design_params": design_params,
        }

        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    # def step(self, action):
    #     """
    #     Execute one environment step

    #     Returns:
    #         observation, reward, terminated, truncated, info
    #     """
    #     self.current_step += 1

    #     # Decode action into design parameters
    #     design_params = self._decode_action(action)

    #     # Validate constraints
    #     if not self._is_valid_design(design_params):
    #         # Penalize invalid designs
    #         reward = -1000.0
    #         terminated = True
    #         info = {"valid": False, "reason": "constraint_violation"}
    #     else:
    #         # Calculate PPAC metrics
    #         ppac = self.ppac_calc.compute(design_params)

    #         # Compute reward
    #         reward = self.reward_fn.compute(ppac)

    #         # Update state
    #         self.state.update(ppac)

    #         # Check termination
    #         terminated = self.current_step >= self.config["max_steps"]

    #         # Tracking
    #         if reward > self.best_reward:
    #             self.best_reward = reward

    #         info = {
    #             "valid": True,
    #             "throughput": ppac["throughput"],
    #             "energy": ppac["energy"],
    #             "cost": ppac["total_cost"],
    #             "ppac": ppac,
    #             "design_params": design_params,
    #         }

    #     truncated = False
    #     observation = self._get_obs()

    #     return observation, reward, terminated, truncated, info

    def _init_state(self) -> Dict:
        """Initialize environment state"""
        return {
            "package_area_budget": self.config["max_package_area"],
            "area_per_chiplet": 0.0,
            "latency_ai2ai": 0.0,
            "latency_hbm2ai": 0.0,
            "energy": 0.0,
            "cost": 0.0,
            "throughput": 0.0,
            "utilization": 0.0,
        }

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector"""
        obs = np.zeros(32, dtype=np.float32)

        # Normalize state values to [0, 1]
        obs[0] = self.state["package_area_budget"] / 1000.0  # Normalize by 1000mm^2
        obs[1] = min(self.state["area_per_chiplet"] / 400.0, 1.0)  # Max 400mm^2
        obs[2] = min(self.state["latency_ai2ai"] / 100.0, 1.0)  # Normalize latency
        obs[3] = min(self.state["latency_hbm2ai"] / 100.0, 1.0)
        obs[4] = min(self.state["energy"] / 10.0, 1.0)  # Normalize energy
        obs[5] = min(self.state["cost"] / 1000.0, 1.0)  # Normalize cost
        obs[6] = min(self.state["throughput"] / 300.0, 1.0)  # Normalize throughput
        obs[7] = self.state["utilization"]

        # Rest are zeros (for future extensibility)
        return obs

    def _decode_action(self, action) -> Dict:
        """Decode MultiDiscrete action into design parameters"""
        num_chiplets = int(action[0]) + 4  # Map [0-124] to [4-128]
        placement_rows = int(action[1]) + 1  # Map [0-15] to [1-16]

        # Calculate columns to match chiplet count
        placement_cols = int(np.ceil(num_chiplets / placement_rows))

        # Decode HBM location bitmask
        hbm_mask = int(action[3])
        hbm_locations = []
        if hbm_mask & 1:
            hbm_locations.append("left")
        if hbm_mask & 2:
            hbm_locations.append("right")
        if hbm_mask & 4:
            hbm_locations.append("top")
        if hbm_mask & 8:
            hbm_locations.append("bottom")
        if hbm_mask & 16:
            hbm_locations.append("middle")

        # Map link count indices to actual values
        link_count_2_5d = (int(action[6]) + 1) * 50  # [50, 5000] step 50
        link_count_3d = (int(action[9]) + 1) * 100  # [100, 10000] step 100

        return {
            "num_chiplets": num_chiplets,
            "placement_rows": placement_rows,
            "placement_cols": placement_cols,
            "num_hbm": int(action[2]) + 1,
            "hbm_locations": hbm_locations,
            "interconnect_2_5d": "EMIB" if action[4] == 1 else "CoWoS",
            "data_rate_2_5d": int(action[5]) + 1,
            "link_count_2_5d": link_count_2_5d,
            "interconnect_3d": "FOVEROS" if action[7] == 1 else "SoIC",
            "data_rate_3d": int(action[8]) + 20,
            "link_count_3d": link_count_3d,
        }

    def _is_valid_design(self, design_params: Dict) -> bool:
        """Check if design satisfies constraints"""
        # Check area constraint
        total_area = (
            design_params["num_chiplets"] * 2
        )  # Rough estimate, will refine in PPAC calc
        if total_area > self.config["max_package_area"]:
            return False

        # Check placement is feasible
        if (
            design_params["placement_rows"] * design_params["placement_cols"]
            < design_params["num_chiplets"]
        ):
            return False

        # Check HBM count
        if design_params["num_hbm"] > len(design_params["hbm_locations"]):
            return False

        return True

    def render(self):
        """Render environment state (optional)"""
        if self.state is None:
            return

        print(f"\n=== Chiplet-Gym State (Step {self.current_step}) ===")
        print(f"Throughput: {self.state['throughput']:.2f}")
        print(f"Energy: {self.state['energy']:.4f}")
        print(f"Cost: {self.state['cost']:.2f}")
        print(f"Best Reward: {self.best_reward:.2f}")
