import numpy as np


class ThermalSolver:
    HRV_EFFICIENCY = 0.8
    RHO_CP_AIR_J_M3K = 1200

    def __init__(
        self,
        thermal_conductance_w_k,
        heat_capacity_j_k,
        ext_air_conductance_w_k,
        ext_ground_conductance_w_k,
        ground_temperature_c,
        areas_m2,
        num_nodes,
    ):
        self.thermal_conductance_w_k = thermal_conductance_w_k
        self.heat_capacity_j_k = heat_capacity_j_k
        self.ext_air_conductance_w_k = ext_air_conductance_w_k
        self.ext_ground_conductance_w_k = ext_ground_conductance_w_k
        self.ground_temperature_c = ground_temperature_c
        self.areas_m2 = areas_m2

        self.temperatures_c = np.full(num_nodes, 21.0)

    def step(self, dt_seconds, out_temperature_c, q_total_w, v_hvac_m3_s):

        q_inter_w = np.dot(self.thermal_conductance_w_k, self.temperatures_c) - (
            np.sum(self.thermal_conductance_w_k, axis=1) * self.temperatures_c
        )

        q_ext_air_w = self.ext_air_conductance_w_k * (
            out_temperature_c - self.temperatures_c
        )

        q_ground_w = self.ext_ground_conductance_w_k * (
            self.ground_temperature_c - self.temperatures_c
        )

        # TODO: fix that
        bypass_active = (out_temperature_c < self.temperatures_c) & (
            self.temperatures_c > 22.0
        )

        effective_eff = np.where(bypass_active, 0.0, self.HRV_EFFICIENCY)

        q_vent_w = (
            v_hvac_m3_s
            * self.RHO_CP_AIR_J_M3K
            * (1 - effective_eff)
            * (out_temperature_c - self.temperatures_c)
        )

        total_q_w = q_inter_w + q_ext_air_w + q_ground_w + q_vent_w + q_total_w

        self.temperatures_c += (total_q_w / self.heat_capacity_j_k) * dt_seconds

        return self.temperatures_c
