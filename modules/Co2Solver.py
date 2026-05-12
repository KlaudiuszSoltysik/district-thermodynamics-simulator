import numpy as np


class Co2Solver:
    def __init__(
        self,
        air_mixing_rate_m3_s,
        volumes_m3,
        infiltration_rate_m3_s,
        num_nodes,
        index_to_id,
    ):
        self.air_mixing_rate_m3_s = air_mixing_rate_m3_s
        self.volumes_m3 = volumes_m3
        self.infiltration_rate_m3_s = infiltration_rate_m3_s
        self.num_nodes = num_nodes
        self.index_to_id = index_to_id

        self.co2_ppm = np.full(self.num_nodes, 500.0)

        self.occupancy_mask = np.zeros((self.num_nodes, 24))

        self.set_occupancy_schedule()

    def set_occupancy_schedule(self):
        # TODO: Implement parsing from local YAML/JSON config later
        return

    def step(self, current_time, dt_seconds, outside_co2_ppm, v_hvac_m3_s):

        current_h = current_time.hour
        current_mask = self.occupancy_mask[:, current_h]

        steps = int(np.ceil(dt_seconds / 60.0))
        micro_dt_s = dt_seconds / steps

        co2_generation_m3_s = (0.015 / 3600.0) * current_mask

        for _ in range(steps):
            co2_mixed_m3_s_ppm = np.dot(self.air_mixing_rate_m3_s, self.co2_ppm) - (
                np.sum(self.air_mixing_rate_m3_s, axis=1) * self.co2_ppm
            )

            co2_infil_m3_s_ppm = self.infiltration_rate_m3_s * (
                outside_co2_ppm - self.co2_ppm
            )

            co2_vented_m3_s_ppm = v_hvac_m3_s * (outside_co2_ppm - self.co2_ppm)

            total_co2_flow_m3_s_ppm = (
                co2_mixed_m3_s_ppm + co2_infil_m3_s_ppm + co2_vented_m3_s_ppm
            )

            self.co2_ppm += (total_co2_flow_m3_s_ppm / self.volumes_m3) * micro_dt_s

            self.co2_ppm += (
                (co2_generation_m3_s / self.volumes_m3) * 1000000.0 * micro_dt_s
            )

        self.co2_ppm = np.maximum(self.co2_ppm, 400.0)

        return np.round(self.co2_ppm).astype(int)
