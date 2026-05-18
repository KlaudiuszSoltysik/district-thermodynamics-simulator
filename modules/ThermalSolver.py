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
    ):
        self.thermal_conductance_w_k = thermal_conductance_w_k
        self.heat_capacity_j_k = heat_capacity_j_k
        self.ext_air_conductance_w_k = ext_air_conductance_w_k
        self.ext_ground_conductance_w_k = ext_ground_conductance_w_k
        self.ground_temperature_c = ground_temperature_c
        self.areas_m2 = areas_m2

        self.T = np.full(len(self.heat_capacity_j_k), 21.0)

    def step(self, dt_seconds, out_temperature_c, q_total_w, v_hvac_m3_s):

        q_inter_w = np.dot(self.thermal_conductance_w_k, self.T) - (
            np.sum(self.thermal_conductance_w_k, axis=1) * self.T
        )

        q_ext_air_w = self.ext_air_conductance_w_k * (out_temperature_c - self.T)

        q_ground_w = self.ext_ground_conductance_w_k * (
            self.ground_temperature_c - self.T
        )

        bypass_active = (out_temperature_c < self.T) & (self.T > 22.0)
        
        effective_eff = np.where(bypass_active, 0.0, self.HRV_EFFICIENCY)

        q_vent_w = (
            v_hvac_m3_s
            * self.RHO_CP_AIR_J_M3K
            * (1.0 - effective_eff)
            * (out_temperature_c - self.T)
        )

        total_q_w = q_inter_w + q_ext_air_w + q_ground_w + q_vent_w + q_total_w

        self.T += (total_q_w / self.heat_capacity_j_k) * dt_seconds

        return self.T
