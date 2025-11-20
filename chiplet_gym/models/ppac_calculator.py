"""
PPAC (Power, Performance, Area, Cost) Calculator
Ported from arch_evaluation.m with analytical models
"""

from typing import Dict

import numpy as np


class PPACCalculator:
    """
    Analytical PPAC model for chiplet-based AI accelerators
    Based on arch_evaluation.m and Chiplet-Gym paper
    """

    def __init__(self, config: Dict):
        self.config = config

        # Workload characteristics (Generic AI workload - average of ResNet50, BERT, etc.)
        # self.ops_per_task = 200e9  # 200 GOps (average across models)
        self.ops_per_task = 4e9  # 4 GOps [ResNet50] (average across models)
        # self.ref_throughput_a100 = 150  # tasks/sec (baseline reference)
        self.ref_throughput_a100 = 191  # tasks/sec (baseline reference)
        self.peak_ops_a100 = 156e12  # 156 TFLOPs

        # Compute scale factor (accounts for non-GEMM ops, mapping efficiency)
        self.scale_factor = self.peak_ops_a100 / (
            self.ref_throughput_a100 * self.ops_per_task
        )

        # Energy constants
        self.joules_per_op_a100 = 400 / (156e12)  # 400W / 156 TFLOPs
        self.joules_per_op_chiplet = 6.8e-13  # Energy per op for chiplet

        # Cost model constants (from MATLAB)
        self.defect_density = 0.09  # 7nm defect density
        self.critical_level = 10
        self.scribe_line = 0.2  # mm
        self.wafer_diameter = 300  # mm
        self.wafer_cost = 9346  # USD
        self.edge_loss = 5  # mm
        self.os_area_scale_factor = 4
        self.package_factor = 2
        self.c4_bump_cost_factor = 0.005
        self.ubump_cost_factor = 0.01
        self.bonding_yield_os = 0.99
        self.area_scale_factor_si = 1.1
        self.defect_density_si = 0.06
        self.bonding_yield_si = 0.99
        self.cost_factor_os = 0.005
        self.cost_3d_penalty = 5  # USD per chiplet for 3D packaging

        # Interconnect parameters (from Table 1 in paper)
        self.interconnect_params = {
            "CoWoS": {
                "bandwidth_per_link": 16e9,
                "latency": 1.0,
                "energy_per_bit": 0.5e-12,
            },
            "EMIB": {
                "bandwidth_per_link": 8e9,
                "latency": 1.5,
                "energy_per_bit": 0.7e-12,
            },
            "SoIC": {
                "bandwidth_per_link": 40e9,
                "latency": 0.5,
                "energy_per_bit": 0.3e-12,
            },
            "FOVEROS": {
                "bandwidth_per_link": 30e9,
                "latency": 0.7,
                "energy_per_bit": 0.4e-12,
            },
        }

    def compute(self, design_params: Dict) -> Dict:
        """
        Compute all PPAC metrics for given design parameters

        Returns:
            dict with keys: throughput, energy, power, area, die_cost,
                           package_cost, total_cost, latency_ai2ai, latency_hbm2ai
        """
        # 1. Performance (Throughput)
        throughput = self._compute_throughput(design_params)

        # 2. Energy
        energy, power = self._compute_energy(design_params, throughput)

        # 3. Area
        total_area, chiplet_area = self._compute_area(design_params)

        # 4. Cost
        die_cost, package_cost, total_cost = self._compute_cost(
            design_params, chiplet_area, total_area
        )

        # 5. Communication latency
        latency_ai2ai = self._compute_ai2ai_latency(design_params)
        latency_hbm2ai = self._compute_hbm2ai_latency(design_params)

        return {
            "throughput": throughput,
            "energy": energy,
            "power": power,
            "area": total_area,
            "chiplet_area": chiplet_area,
            "die_cost": die_cost,
            "package_cost": package_cost,
            "total_cost": total_cost,
            "latency_ai2ai": latency_ai2ai,
            "latency_hbm2ai": latency_hbm2ai,
        }

    def _compute_throughput(self, design_params: Dict) -> float:
        """
        Compute throughput (tasks/sec) based on chiplet parallelism
        Formula: throughput = peak_ops / (scale_factor * ops_per_task)
        """
        num_chiplets = design_params["num_chiplets"]

        # Peak compute per chiplet (assume 2.5 TFLOPs per chiplet based on paper)
        peak_ops_per_chiplet = 2.5e12  # TFLOPs
        peak_ops_total = num_chiplets * peak_ops_per_chiplet

        # Account for communication overhead (more chiplets = more comm overhead)
        comm_efficiency = self._compute_comm_efficiency(design_params)

        # Effective throughput
        throughput = (peak_ops_total * comm_efficiency) / (
            self.scale_factor * self.ops_per_task
        )

        return throughput

    def _compute_comm_efficiency(self, design_params: Dict) -> float:
        """
        Compute communication efficiency based on interconnect topology
        Efficiency decreases with more chiplets and worse interconnect
        """
        num_chiplets = design_params["num_chiplets"]
        interconnect_2_5d = design_params["interconnect_2_5d"]
        interconnect_3d = design_params["interconnect_3d"]
        link_count_2_5d = design_params["link_count_2_5d"]
        link_count_3d = design_params["link_count_3d"]

        # Get interconnect bandwidth
        bw_2_5d = self.interconnect_params[interconnect_2_5d]["bandwidth_per_link"]
        bw_3d = self.interconnect_params[interconnect_3d]["bandwidth_per_link"]

        # Total bandwidth
        total_bw = link_count_2_5d * bw_2_5d + link_count_3d * bw_3d

        # Required bandwidth (heuristic: scales with chiplet count)
        required_bw = num_chiplets * 100e9  # 100 Gbps per chiplet (rough estimate)

        # Efficiency = min(1, total_bw / required_bw)
        efficiency = min(1.0, total_bw / required_bw)

        # Penalty for large chiplet count (network congestion)
        congestion_penalty = 1.0 / (1.0 + 0.001 * num_chiplets)

        return efficiency * congestion_penalty

    def _compute_energy(self, design_params: Dict, throughput: float) -> tuple:
        """
        Compute energy per task and power
        Returns: (energy_per_task, power)
        """
        # Compute energy (based on actual ops)
        compute_energy = self.joules_per_op_chiplet * self.ops_per_task

        # Communication energy
        interconnect_2_5d = design_params["interconnect_2_5d"]
        interconnect_3d = design_params["interconnect_3d"]

        # Data movement: 5% of ops require inter-chiplet communication
        data_fraction = 0.05
        bytes_per_op = 4  # FP32
        data_per_task = self.ops_per_task * data_fraction * bytes_per_op  # bytes

        energy_per_bit_2_5d = self.interconnect_params[interconnect_2_5d][
            "energy_per_bit"
        ]
        energy_per_bit_3d = self.interconnect_params[interconnect_3d]["energy_per_bit"]

        # Weighted average (70% 2.5D horizontal, 30% 3D vertical)
        avg_energy_per_bit = 0.7 * energy_per_bit_2_5d + 0.3 * energy_per_bit_3d

        # Communication energy (convert bytes to bits)
        comm_energy = data_per_task * 8 * avg_energy_per_bit

        # Total energy per task (Joules)
        total_energy_per_task = compute_energy + comm_energy

        # Power (Watts) = Energy/task × Tasks/sec
        power = total_energy_per_task * throughput

        return total_energy_per_task, power

    # def _compute_energy(self, design_params: Dict, throughput: float) -> tuple:
    #     """
    #     Compute energy per task and power
    #     Returns: (energy_per_task, power)
    #     """
    #     num_chiplets = design_params["num_chiplets"]

    #     # Compute energy per operation (FIX: Remove scale_factor)
    #     # Energy should be based on actual ops executed, not scaled ops
    #     compute_energy = self.joules_per_op_chiplet * self.ops_per_task

    #     # Communication energy (2.5D + 3D interconnects)
    #     interconnect_2_5d = design_params["interconnect_2_5d"]
    #     interconnect_3d = design_params["interconnect_3d"]

    #     # Assume 10% of data needs inter-chiplet communication
    #     data_per_task = self.ops_per_task * 0.1 * 4  # 4 bytes per op (FP32)

    #     energy_per_bit_2_5d = self.interconnect_params[interconnect_2_5d][
    #         "energy_per_bit"
    #     ]
    #     energy_per_bit_3d = self.interconnect_params[interconnect_3d]["energy_per_bit"]

    #     # Half comm through 2.5D, half through 3D (heuristic)
    #     comm_energy = (
    #         data_per_task * 8 * (0.5 * energy_per_bit_2_5d + 0.5 * energy_per_bit_3d)
    #     )

    #     # Total energy per task (in Joules)
    #     total_energy_per_task = compute_energy + comm_energy

    #     # Power = Energy per task × Throughput (tasks/sec) = Watts
    #     power = total_energy_per_task * throughput

    #     return total_energy_per_task, power

    # def _compute_energy(self, design_params: Dict, throughput: float) -> tuple:
    #     """
    #     Compute energy per task and power
    #     Returns: (energy_per_task, power)
    #     """
    #     num_chiplets = design_params["num_chiplets"]

    #     # Compute energy
    #     compute_energy = (
    #         self.joules_per_op_chiplet * self.ops_per_task * self.scale_factor
    #     )

    #     # Communication energy (2.5D + 3D interconnects)
    #     interconnect_2_5d = design_params["interconnect_2_5d"]
    #     interconnect_3d = design_params["interconnect_3d"]

    #     # Assume 10% of data needs inter-chiplet communication
    #     data_per_task = self.ops_per_task * 0.1 * 4  # 4 bytes per op

    #     energy_per_bit_2_5d = self.interconnect_params[interconnect_2_5d][
    #         "energy_per_bit"
    #     ]
    #     energy_per_bit_3d = self.interconnect_params[interconnect_3d]["energy_per_bit"]

    #     # Half comm through 2.5D, half through 3D (heuristic)
    #     comm_energy = (
    #         data_per_task * 8 * (0.5 * energy_per_bit_2_5d + 0.5 * energy_per_bit_3d)
    #     )

    #     total_energy_per_task = compute_energy + comm_energy

    #     # Power = Energy * Throughput
    #     power = total_energy_per_task * throughput

    #     return total_energy_per_task, power

    def _compute_area(self, design_params: Dict) -> tuple:
        """
        Compute total area and per-chiplet area
        Returns: (total_area, chiplet_area)
        """
        num_chiplets = design_params["num_chiplets"]
        num_hbm = design_params["num_hbm"]

        # Assume 800 mm^2 total compute area (from MATLAB)
        total_compute_area = 800  # mm^2
        chiplet_area = total_compute_area / num_chiplets

        # HBM area (each HBM ~100 mm^2)
        hbm_area = num_hbm * 100

        # Package area (includes interposer, substrate)
        package_overhead = 1.5  # 50% overhead
        total_area = (total_compute_area + hbm_area) * package_overhead

        return total_area, chiplet_area

    def _compute_cost(
        self, design_params: Dict, chiplet_area: float, total_area: float
    ) -> tuple:
        """
        Compute die cost, package cost, and total cost
        Ported from MATLAB arch_evaluation.m
        Returns: (die_cost, package_cost, total_cost)
        """
        num_chiplets = design_params["num_chiplets"]

        # Die cost calculation
        chiplet_area_f = (
            chiplet_area
            + 2 * self.scribe_line * np.sqrt(chiplet_area)
            + self.scribe_line**2
        )

        N_die_total = np.pi * (
            (self.wafer_diameter / 2 - self.edge_loss) ** 2
        ) / chiplet_area_f - np.pi * (
            self.wafer_diameter - 2 * self.edge_loss
        ) / np.sqrt(2 * chiplet_area_f)

        die_yield = (
            1 + (self.defect_density * chiplet_area) / (100 * self.critical_level)
        ) ** (-self.critical_level)

        N_KGD = N_die_total * die_yield

        cost_raw_die = self.wafer_cost / N_die_total
        cost_KGD = self.wafer_cost / N_KGD
        cost_defect_die = cost_KGD - cost_raw_die
        cost_die_RE = cost_raw_die + cost_defect_die

        # Package cost calculation
        die_area_tot = 900  # mm^2 (from MATLAB)
        interposer_area = die_area_tot * self.area_scale_factor_si
        package_area = interposer_area * self.os_area_scale_factor

        package_yield = (
            1 + (self.defect_density_si * interposer_area) / (100 * self.critical_level)
        ) ** (-self.critical_level)

        interposer_area_f = (
            interposer_area
            + 2 * self.scribe_line * np.sqrt(interposer_area)
            + self.scribe_line**2
        )

        N_package_total = np.pi * (
            (self.wafer_diameter / 2 - self.edge_loss) ** 2
        ) / interposer_area_f - np.pi * (
            self.wafer_diameter - 2 * self.edge_loss
        ) / np.sqrt(2 * interposer_area_f)

        cost_interposer = (
            self.wafer_cost / N_package_total
        ) + interposer_area * self.c4_bump_cost_factor
        cost_substrate = package_area * self.cost_factor_os
        cost_raw_package = cost_interposer + cost_substrate

        # Chip integration cost
        cost_raw_chips = (
            cost_raw_die * num_chiplets
            + chiplet_area * self.ubump_cost_factor
            + num_chiplets * self.cost_3d_penalty
        )
        cost_defect_chips = cost_defect_die * num_chiplets

        bonding_yield = self.bonding_yield_si**num_chiplets * self.bonding_yield_si**2

        cost_defect_package = cost_interposer * (
            1 / (package_yield * bonding_yield * self.bonding_yield_os) - 1
        ) + cost_substrate * (1 / self.bonding_yield_os - 1)

        cost_wasted_chips = (cost_raw_chips + cost_defect_chips) * (
            1 / (bonding_yield * self.bonding_yield_os)
        ) - 1

        cost_RE_package = (
            cost_raw_chips
            + cost_defect_chips
            + cost_raw_package
            + cost_defect_package
            + cost_wasted_chips
        )

        total_cost = cost_RE_package + cost_die_RE

        return cost_die_RE, cost_RE_package, total_cost

    def _compute_ai2ai_latency(self, design_params: Dict) -> float:
        """Compute AI chiplet to AI chiplet latency (ns)"""
        interconnect_2_5d = design_params["interconnect_2_5d"]
        latency = self.interconnect_params[interconnect_2_5d]["latency"]
        return latency

    def _compute_hbm2ai_latency(self, design_params: Dict) -> float:
        """Compute HBM to AI chiplet latency (ns)"""
        interconnect_3d = design_params["interconnect_3d"]
        latency = self.interconnect_params[interconnect_3d]["latency"]
        return latency
