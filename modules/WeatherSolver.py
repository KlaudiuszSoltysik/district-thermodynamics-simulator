import math

import numpy as np


class WeatherSolver:
    RHO_CP_AIR_J_M3K = 1200
    H_EXTERNAL_W_M2K = 25

    def __init__(self, external_connections, standards, num_nodes):
        self.connections = external_connections
        self.standards = standards
        self.num_nodes = num_nodes

    def calculate_environmental_gains(
        self,
        sun_radiation_w_m2,
        sun_altitude_deg,
        sun_azimuth_deg,
        wind_speed_m_s,
        wind_direction_deg,
        temperature_ext_c,
        temperature_rooms_c,
    ):
        q_env_w = np.zeros(self.num_nodes)

        if sun_altitude_deg > 0:
            for conn in self.connections:
                azimuth_diff_rad = math.radians(sun_azimuth_deg - conn["azimuth"])
                el_rad = math.radians(sun_altitude_deg)
                tilt_rad = math.radians(conn["tilt"])

                cos_theta = math.sin(el_rad) * math.cos(tilt_rad) + math.cos(
                    el_rad
                ) * math.sin(tilt_rad) * math.cos(azimuth_diff_rad)

                if cos_theta > 0:
                    room_idx = conn["room_idx"]
                    win_area_sum_m2 = 0

                    for window in conn.get("windows", []):
                        q_env_w[room_idx] += (
                            window["area_m2"]
                            * sun_radiation_w_m2
                            * window["shgc"]
                            * cos_theta
                        )
                        win_area_sum_m2 += window["area_m2"]

                    wall_net_area_m2 = conn["area_gross_m2"] - win_area_sum_m2

                    q_env_w[room_idx] += (
                        wall_net_area_m2
                        * sun_radiation_w_m2
                        * conn["absorptance"]
                        * cos_theta
                        * (conn["u_value_w_m2k"] / self.H_EXTERNAL_W_M2K)
                    )

        for conn in self.connections:
            room_idx = conn["room_idx"]

            if conn["tilt"] < 10:
                exposure = 1
            else:
                wind_az_diff_rad = math.radians(wind_direction_deg - conn["azimuth"])
                exposure = (math.cos(wind_az_diff_rad) + 1) / 2

            infiltration_m3_s = (
                conn["ach_wind_coef"] * wind_speed_m_s * exposure
            ) / 3600

            q_env_w[room_idx] += (
                infiltration_m3_s
                * conn["volume_m3"]
                * self.RHO_CP_AIR_J_M3K
                * (temperature_ext_c - temperature_rooms_c[room_idx])
            )

        return q_env_w
