import numpy as np


class ThermalSolver:
    HRV_EFFICIENCY = 0.8
    RHO_CP_AIR = 1200

    def __init__(self, G_temp, C, G_ext_air, G_ext_ground, T_ground, A):
        self.G_temp = G_temp
        self.C = C
        self.G_ext_air = G_ext_air
        self.G_ext_ground = G_ext_ground
        self.T_ground = T_ground
        self.A = A

        self.T = np.full(len(C), 21.0)

    def step(self, dt, T_outside, q_hvac, v_hvac):
        Q_inter = np.dot(self.G_temp, self.T) - (np.sum(self.G_temp, axis=1) * self.T)

        Q_air = self.G_ext_air * (T_outside - self.T)

        Q_ground = self.G_ext_ground * (self.T_ground - self.T)

        Q_vent = (
            v_hvac
            * self.RHO_CP_AIR
            * (1.0 - self.HRV_EFFICIENCY)
            * (T_outside - self.T)
        )

        total_Q = Q_inter + Q_air + Q_ground + Q_vent + q_hvac

        self.T += (total_Q / self.C) * dt

        return self.T
