import numpy as np
import pandas as pd


class RoughController:
    HORIZON_HOURS = 24
    TEMPERATURE_TOLERANCE_C = 0.5
    MAX_VENT_POWER_M3_S = 0.02
    HRV_EFFICIENCY = 0.8
    RHO_CP_AIR_J_M3K = 1200

    def __init__(
        self,
        heat_pump,
        thermal_solver,
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
        self.load_schedules(schedules_data)

        self.heat_capacity_j_k = np.copy(thermal_solver.heat_capacity_j_k)
        self.ext_air_conductance_w_k = np.copy(thermal_solver.ext_air_conductance_w_k)
        self.ext_ground_conductance_w_k = np.copy(
            thermal_solver.ext_ground_conductance_w_k
        )
        self.ground_temperature_c = thermal_solver.ground_temperature_c

        self.thermal_conductance_w_k = np.copy(thermal_solver.thermal_conductance_w_k)

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

    def predict_baseline(
        self,
        simulation_time,
        dt_seconds,
        current_temperatures_c,
        weather_service,
        energy_service,
        pv_farm,
    ):
        steps = int(self.HORIZON_HOURS * 3600 / dt_seconds)
        sim_temperatures_c = np.copy(current_temperatures_c)
        sim_time = simulation_time

        baseline_load_kw = np.zeros(steps)
        baseline_pv_kw = np.zeros(steps)

        for i in range(steps):
            current_h = sim_time.hour
            target_temperatures_c = self.target_temp_24h_c[:, current_h]

            weather = weather_service.get_weather(sim_time)
            energy_costs = energy_service.get_effective_costs(
                sim_time, pv_farm, self.heat_pump, weather
            )
            out_temperature_c = weather.temperature_c

            step_q_w = np.zeros(self.num_nodes)
            step_v_m3_s = np.full(self.num_nodes, self.MAX_VENT_POWER_M3_S * 0.5)

            for j in range(self.num_nodes):
                target_temp = target_temperatures_c[j]

                if not np.isnan(target_temp):
                    if sim_temperatures_c[j] < (
                        target_temp - self.TEMPERATURE_TOLERANCE_C
                    ):
                        step_q_w[j] = self.max_heating_power_w[j]
                    elif sim_temperatures_c[j] > (
                        target_temp + self.TEMPERATURE_TOLERANCE_C
                    ):
                        step_q_w[j] = -self.max_cooling_power_w[j]

                if sim_temperatures_c[j] < 16.0:
                    step_q_w[j] = self.max_heating_power_w[j]

            q_heating_w = np.maximum(0, step_q_w)
            q_cooling_w = np.maximum(0, -step_q_w)

            heat_elec_kw = (np.sum(q_heating_w) / energy_costs.cop_heating) / 1000.0
            cool_elec_kw = (np.sum(q_cooling_w) / energy_costs.cop_cooling) / 1000.0
            vent_elec_kw = np.sum(step_v_m3_s) * 1.0

            baseline_load_kw[i] = heat_elec_kw + cool_elec_kw + vent_elec_kw
            baseline_pv_kw[i] = energy_costs.pv_yield_kw

            q_inter_w = np.dot(self.thermal_conductance_w_k, sim_temperatures_c) - (
                np.sum(self.thermal_conductance_w_k, axis=1) * sim_temperatures_c
            )

            q_ext_air_w = self.ext_air_conductance_w_k * (
                out_temperature_c - sim_temperatures_c
            )
            q_ground_w = self.ext_ground_conductance_w_k * (
                self.ground_temperature_c - sim_temperatures_c
            )

            q_solar_w = (weather.sun_radiation_w_m2 * 0.1) * np.ones(self.num_nodes)

            safe_target = np.where(
                np.isnan(target_temperatures_c), 21.0, target_temperatures_c
            )
            bypass_active = (out_temperature_c < sim_temperatures_c) & (
                sim_temperatures_c > safe_target
            )
            effective_eff = np.where(bypass_active, 0.0, self.HRV_EFFICIENCY)

            q_vent_w = (
                step_v_m3_s
                * self.RHO_CP_AIR_J_M3K
                * (1 - effective_eff)
                * (out_temperature_c - sim_temperatures_c)
            )

            total_q_w = (
                q_inter_w + q_ext_air_w + q_ground_w + q_solar_w + q_vent_w + step_q_w
            )
            sim_temperatures_c += (total_q_w / self.heat_capacity_j_k) * dt_seconds

            sim_time += pd.Timedelta(seconds=dt_seconds)

        return baseline_load_kw, baseline_pv_kw
