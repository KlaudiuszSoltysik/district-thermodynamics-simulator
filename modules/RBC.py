import numpy as np


class RBC:
    TEMPERATURE_TOLERANCE_C = 0.2
    MAX_VENT_POWER_M3_S = 0.02

    def __init__(
        self,
        heat_pump,
        num_nodes,
        district_id_dict,
        schedules_data,
    ):
        self.heat_pump = heat_pump
        self.num_nodes = num_nodes
        self.district_id_dict = district_id_dict

        self.max_heating_power_w = heat_pump.max_heating_power_w
        self.max_cooling_power_w = heat_pump.max_cooling_power_w
        self.target_temp_24h_c = np.full((self.num_nodes, 24), 21.0, dtype=float)
        self.max_co2_24h_ppm = np.full((self.num_nodes, 24), 1000.0, dtype=float)
        self.is_occupied_24h = np.zeros((self.num_nodes, 24), dtype=float)

        self.load_schedules(schedules_data)

    def load_schedules(self, schedules_data):
        for idx in range(self.num_nodes):
            room_key = self.district_id_dict[idx]

            if room_key in schedules_data:
                schedule = schedules_data[room_key]

                temps = schedule.get("target_temp_c")
                if temps and len(temps) == 24:
                    temps_array = np.array(
                        [np.nan if t is None else float(t) for t in temps]
                    )
                    self.target_temp_24h_c[idx, :] = temps_array

                co2_limits = schedule.get("max_co2_ppm")
                if co2_limits:
                    self.max_co2_24h_ppm[idx, :] = np.array(co2_limits, dtype=float)

                occupied = schedule.get("is_occupied")
                if occupied and len(occupied) == 24:
                    self.is_occupied_24h[idx, :] = np.array(occupied, dtype=float)

    def step(
        self,
        current_time,
        dt_seconds,
        weather_solver,
        thermal_solver,
        co2_solver,
        weather_service,
        energy_service,
    ):
        current_h = current_time.hour

        current_q_w = np.zeros(self.num_nodes)
        current_v_m3_s = np.zeros(self.num_nodes)

        current_temperatures_c = thermal_solver.temperatures_c
        current_co2_ppm = co2_solver.co2_ppm

        target_temperatures_c = self.target_temp_24h_c[:, current_h]
        max_co2_ppm = self.max_co2_24h_ppm[:, current_h]

        for i in range(self.num_nodes):
            target_temp = target_temperatures_c[i]

            if not np.isnan(target_temp):
                if current_temperatures_c[i] < (
                    target_temp - self.TEMPERATURE_TOLERANCE_C
                ):
                    current_q_w[i] = self.max_heating_power_w[i]
                elif current_temperatures_c[i] > (
                    target_temp + self.TEMPERATURE_TOLERANCE_C
                ):
                    current_q_w[i] = -self.max_cooling_power_w[i]

            if current_temperatures_c[i] < 16.0:
                current_q_w[i] = self.max_heating_power_w[i]

            if current_co2_ppm[i] > max_co2_ppm[i]:
                current_v_m3_s[i] = self.MAX_VENT_POWER_M3_S

        forecast_data = []

        return current_q_w, current_v_m3_s, forecast_data
