import numpy as np


# TODO: Make sure its calculated properly
class Co2Solver:
    CO2_GENERATION_M3_H = 0.025
    PERSON_PER_M3 = 1 / (30 * 2.5)

    def __init__(
        self,
        air_mixing_rate_m3_s,
        volumes_m3,
        infiltration_rate_m3_s,
        num_nodes,
        index_to_id,
        schedules_data,
    ):
        self.air_mixing_rate_m3_s = air_mixing_rate_m3_s
        self.infiltration_rate_m3_s = infiltration_rate_m3_s

        self.volumes_m3 = volumes_m3
        self.num_nodes = num_nodes
        self.index_to_id = index_to_id

        self.co2_ppm = np.full(self.num_nodes, 800.0)

        self.occupancy_mask = np.zeros((self.num_nodes, 24))

        self.base_people_in_room = self.volumes_m3 * self.PERSON_PER_M3

        self.set_occupancy_schedule(schedules_data)

    def set_occupancy_schedule(self, schedules_data):
        for idx in range(self.num_nodes):
            room_key = self.index_to_id[idx]

            if room_key in schedules_data:
                occ_24h = schedules_data[room_key].get("is_occupied", [0] * 24)
                self.occupancy_mask[idx, :] = occ_24h

    def step(self, current_time, dt_seconds, outside_co2_ppm, v_hvac_m3_s):
        current_h = current_time.hour
        current_mask = self.occupancy_mask[:, current_h]

        flows_out_m3_s = (
            np.sum(self.air_mixing_rate_m3_s, axis=1)
            + self.infiltration_rate_m3_s
            + v_hvac_m3_s
        )

        ach_per_second = np.where(
            self.volumes_m3 > 0, flows_out_m3_s / self.volumes_m3, 0
        )

        max_ach_per_second = np.max(ach_per_second)

        safe_dt = 0.5 / max_ach_per_second

        steps = int(np.ceil(dt_seconds / safe_dt))
        micro_dt_s = dt_seconds / steps

        active_people = self.base_people_in_room * current_mask

        co2_generation_m3_s = (self.CO2_GENERATION_M3_H / 3600) * active_people

        for _ in range(steps):
            co2_mixed_m3_s_ppm = np.dot(self.air_mixing_rate_m3_s, self.co2_ppm) - (
                np.sum(self.air_mixing_rate_m3_s, axis=1) * self.co2_ppm
            )

            co2_infiltration_m3_s_ppm = self.infiltration_rate_m3_s * (
                outside_co2_ppm - self.co2_ppm
            )

            co2_vented_m3_s_ppm = v_hvac_m3_s * (outside_co2_ppm - self.co2_ppm)

            total_co2_flow_m3_s_ppm = (
                co2_mixed_m3_s_ppm + co2_infiltration_m3_s_ppm + co2_vented_m3_s_ppm
            )

            self.co2_ppm += (total_co2_flow_m3_s_ppm / self.volumes_m3) * micro_dt_s

            self.co2_ppm += (
                (co2_generation_m3_s / self.volumes_m3) * 1000000 * micro_dt_s
            )

        return np.round(self.co2_ppm).astype(int)
