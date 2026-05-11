import math

import numpy as np


class WeatherSolver:
    RHO_CP_AIR = 1200
    H_EXTERNAL = 25.0

    def __init__(self, external_connections, standards, N):
        self.connections = external_connections
        self.standards = standards
        self.N = N

    def calculate_environmental_gains(
        self,
        sun_rad,
        sun_altitude,
        sun_azimuth,
        wind_speed,
        wind_direction,
        temperature_ext,
        temperature_rooms,
    ):
        q_net = np.zeros(self.N)

        if sun_altitude > 0:
            for connection in self.connections:
                azimuth_diff = math.radians(sun_azimuth - connection["azimuth"])
                el_rad = math.radians(sun_altitude)
                tilt_rad = math.radians(connection["tilt"])

                cos_theta = math.sin(el_rad) * math.cos(tilt_rad) + math.cos(
                    el_rad
                ) * math.sin(tilt_rad) * math.cos(azimuth_diff)

                if cos_theta > 0:
                    room_idx = connection["room_idx"]
                    win_area_sum = 0

                    for window in connection["windows"]:
                        q_net[room_idx] += (
                            window["area"] * sun_rad * window["shgc"] * cos_theta
                        )
                        win_area_sum += window["area"]

                    wall_net_area = connection["area_gross"] - win_area_sum

                    q_net[room_idx] += (
                        wall_net_area
                        * sun_rad
                        * connection["absorptance"]
                        * cos_theta
                        * (connection["u_value"] / self.H_EXTERNAL)
                    )

        for connection in self.connections:
            room_idx = connection["room_idx"]

            if connection["tilt"] < 10:
                exposure = 1.0
            else:
                wind_az_diff = math.radians(wind_direction - connection["azimuth"])
                exposure = (math.cos(wind_az_diff) + 1) / 2

            n_wind = (connection["ach_wind_coef"] * wind_speed * exposure) / 3600

            q_net[room_idx] += (
                n_wind
                * connection["volume"]
                * self.RHO_CP_AIR
                * (temperature_ext - temperature_rooms[room_idx])
            )

        return q_net
