import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize


class MPC:
    TEMPERATURE_TOLERANCE_C = 1.0
    CO2_GENERATION_M3_H_PER_PERSON = 0.025
    RECALCULATION_INTERVAL_STEPS = 12
    HORIZON_HOURS = 6
    STEPS_PER_HOUR = 1
    MAX_VENT_POWER_M3_S = 0.02
    CONTRACTED_POWER_KW = 15
    PERSON_PER_M3 = 1 / (30 * 2.5)
    HRV_EFFICIENCY = 0.8

    def __init__(
        self,
        pv_farm,
        heat_pump,
        bess,
        num_nodes,
        district_id_dict,
        schedules_data,
    ):
        self.control_steps = self.HORIZON_HOURS * self.STEPS_PER_HOUR

        self.cached_q_plan_w = None
        self.cached_v_plan_m3_s = None
        self.cached_bess_plan_kw = None
        self.steps_since_last_recalc = 0

        self.pv_farm = pv_farm
        self.heat_pump = heat_pump
        self.bess = bess
        self.num_nodes = num_nodes
        self.district_id_dict = district_id_dict

        self.max_heating_power_w = heat_pump.max_heating_power_w
        self.max_cooling_power_w = heat_pump.max_cooling_power_w

        self.target_temp_24h_c = np.full((self.num_nodes, 24), 21.0, dtype=float)
        self.max_co2_24h_ppm = np.full((self.num_nodes, 24), 1000)
        self.is_occupied_24h = np.zeros((self.num_nodes, 24))

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
                    self.max_co2_24h_ppm[idx, :] = co2_limits

                occupied = schedule.get("is_occupied")
                if occupied and len(occupied) == 24:
                    self.is_occupied_24h[idx, :] = occupied

    def get_target_trajectories(self, start_time, dt_seconds, horizon_steps):
        temperature_target_horizon_c = np.zeros((horizon_steps, self.num_nodes))
        co2_max_horizon_ppm = np.zeros((horizon_steps, self.num_nodes))
        is_occupied_horizon = np.zeros((horizon_steps, self.num_nodes))

        future_time = start_time

        for k in range(horizon_steps):
            h0 = future_time.hour

            temperature_target_horizon_c[k, :] = self.target_temp_24h_c[:, h0]
            co2_max_horizon_ppm[k, :] = self.max_co2_24h_ppm[:, h0]
            is_occupied_horizon[k, :] = self.is_occupied_24h[:, h0]

            future_time += pd.Timedelta(seconds=dt_seconds)

        return temperature_target_horizon_c, co2_max_horizon_ppm, is_occupied_horizon

    def step(
        self,
        current_time,
        dt_seconds,
        weather_solver,
        thermal_solver,
        co2_solver,
        weather_service,
        energy_service,
        current_bess_soc,
    ):
        horizon_steps = int((self.HORIZON_HOURS * 3600) / dt_seconds)

        needs_recalc = False
        if (
            self.cached_q_plan_w is None
            or self.cached_v_plan_m3_s is None
            or self.cached_bess_plan_kw is None
            or self.steps_since_last_recalc >= self.RECALCULATION_INTERVAL_STEPS
        ):
            needs_recalc = True

        forecast_data = None

        if needs_recalc:
            if (
                self.cached_q_plan_w is not None
                and self.cached_v_plan_m3_s is not None
                and self.steps_since_last_recalc < len(self.cached_q_plan_w)
            ):
                current_q_w = self.cached_q_plan_w[self.steps_since_last_recalc, :]
                current_v_m3_s = self.cached_v_plan_m3_s[
                    self.steps_since_last_recalc, :
                ]
            else:
                current_q_w = np.zeros(self.num_nodes)
                current_v_m3_s = np.zeros(self.num_nodes)

            if (
                self.cached_bess_plan_kw is not None
                and self.steps_since_last_recalc < len(self.cached_bess_plan_kw)
            ):
                current_bess_kw = self.cached_bess_plan_kw[self.steps_since_last_recalc]
            else:
                current_bess_kw = 0.0

            temperature_out_forecast_c = np.zeros(horizon_steps)
            co2_out_forecast_ppm = np.zeros(horizon_steps)
            q_env_forecast_w = np.zeros((horizon_steps, self.num_nodes))
            elec_cost_forecast_eur_mwh = np.zeros(horizon_steps)
            res_yield_forecast_kw = np.zeros(horizon_steps)
            cop_heat_forecast = np.zeros(horizon_steps)
            cop_cool_forecast = np.zeros(horizon_steps)

            future_time = current_time
            temperatures_frozen_for_prediction_c = np.copy(
                thermal_solver.temperatures_c
            )

            for i in range(horizon_steps):
                weather = weather_service.get_weather(future_time)
                temperature_out_forecast_c[i] = weather.temperature_c
                co2_out_forecast_ppm[i] = weather.co2_ppm

                q_env = weather_solver.calculate_environmental_gains(
                    weather.sun_radiation_w_m2,
                    weather.sun_altitude_deg,
                    weather.sun_azimuth_deg,
                    weather.wind_speed_m_s,
                    weather.wind_direction_deg,
                    weather.temperature_c,
                    temperatures_frozen_for_prediction_c,
                )
                q_env_forecast_w[i, :] = q_env

                energy_costs = energy_service.get_effective_costs(
                    future_time, self.pv_farm, self.heat_pump, weather
                )
                elec_cost_forecast_eur_mwh[i] = (
                    energy_costs.electricity_price_eur_per_mwh
                )
                res_yield_forecast_kw[i] = energy_costs.pv_yield_kw
                cop_heat_forecast[i] = energy_costs.cop_heating
                cop_cool_forecast[i] = energy_costs.cop_cooling

                future_time += pd.Timedelta(seconds=dt_seconds)

            temperature_target_horizon_c, co2_max_horizon_ppm, is_occupied_horizon = (
                self.get_target_trajectories(current_time, dt_seconds, horizon_steps)
            )

            num_decisions_q = self.control_steps * self.num_nodes
            num_decisions_v = self.control_steps * self.num_nodes
            num_decisions_bess = self.control_steps

            bounds_q_perc = [(-100, 100) for _ in range(num_decisions_q)]
            bounds_v_perc = [(0, 100) for _ in range(num_decisions_v)]
            bounds_bess_perc = [(-100, 100) for _ in range(num_decisions_bess)]

            bounds_perc = bounds_q_perc + bounds_v_perc + bounds_bess_perc

            initial_guess = np.zeros(
                num_decisions_q + num_decisions_v + num_decisions_bess
            )

            base_people = co2_solver.volumes_m3 * self.PERSON_PER_M3
            co2_generation_rates_m3_s = (
                self.CO2_GENERATION_M3_H_PER_PERSON / 3600
            ) * base_people

            args = (
                self.control_steps,
                self.num_nodes,
                np.asarray(self.max_heating_power_w, dtype=float),
                np.asarray(self.max_cooling_power_w, dtype=float),
                float(self.MAX_VENT_POWER_M3_S),
                float(self.bess.max_power_kw),
                float(self.bess.capacity_kwh),
                float(self.bess.efficiency),
                float(self.bess.min_soc),
                float(self.bess.max_soc),
                float(current_bess_soc),
                float(current_bess_kw),
                co2_generation_rates_m3_s.astype(float),
                float(self.TEMPERATURE_TOLERANCE_C),
                float(dt_seconds),
                int(horizon_steps),
                thermal_solver.temperatures_c.astype(float),
                co2_solver.co2_ppm.astype(float),
                np.asarray(current_q_w, dtype=float),
                np.asarray(current_v_m3_s, dtype=float),
                float(self.CONTRACTED_POWER_KW),
                float(self.HRV_EFFICIENCY),
                temperature_target_horizon_c.astype(float),
                co2_max_horizon_ppm.astype(float),
                is_occupied_horizon.astype(float),
                temperature_out_forecast_c.astype(float),
                co2_out_forecast_ppm.astype(float),
                q_env_forecast_w.astype(float),
                elec_cost_forecast_eur_mwh.astype(float),
                res_yield_forecast_kw.astype(float),
                cop_heat_forecast.astype(float),
                cop_cool_forecast.astype(float),
                co2_solver.air_mixing_rate_m3_s.astype(float),
                co2_solver.infiltration_rate_m3_s.astype(float),
                co2_solver.volumes_m3.astype(float),
                thermal_solver.thermal_conductance_w_k.astype(float),
                thermal_solver.ext_air_conductance_w_k.astype(float),
                thermal_solver.ext_ground_conductance_w_k.astype(float),
                float(thermal_solver.ground_temperature_c),
                thermal_solver.heat_capacity_j_k.astype(float),
            )

            res = minimize(
                mpc_cost_function,
                initial_guess,
                args=args,
                method="L-BFGS-B",
                bounds=bounds_perc,
                options={"maxiter": 10, "ftol": 1e-2, "disp": False},
            )

            idx_q_end = self.control_steps * self.num_nodes
            idx_v_end = idx_q_end + (self.control_steps * self.num_nodes)

            optimal_q_percent = res.x[:idx_q_end].reshape(
                (self.control_steps, self.num_nodes)
            )
            optimal_v_percent = res.x[idx_q_end:idx_v_end].reshape(
                (self.control_steps, self.num_nodes)
            )
            optimal_bess_percent = res.x[idx_v_end:]

            optimal_q_w = np.where(
                optimal_q_percent >= 0,
                (optimal_q_percent / 100) * self.max_heating_power_w,
                (optimal_q_percent / 100) * self.max_cooling_power_w,
            )
            optimal_v_m3_s = (optimal_v_percent / 100) * self.MAX_VENT_POWER_M3_S

            optimal_bess_kw = (optimal_bess_percent / 100.0) * self.bess.max_power_kw

            block_size = horizon_steps // self.control_steps

            self.cached_q_plan_w = np.repeat(optimal_q_w, block_size, axis=0)
            self.cached_v_plan_m3_s = np.repeat(optimal_v_m3_s, block_size, axis=0)
            self.cached_bess_plan_kw = np.repeat(optimal_bess_kw, block_size, axis=0)

            forecast_data = []
            future_time = current_time

            for i in range(horizon_steps):
                q_dict = {
                    self.district_id_dict[i]: float(self.cached_q_plan_w[i, i])
                    for i in range(self.num_nodes)
                }
                v_dict = {
                    self.district_id_dict[i]: float(self.cached_v_plan_m3_s[i, i])
                    for i in range(self.num_nodes)
                }

                temperature_target_dict = {}
                for i in range(self.num_nodes):
                    val = temperature_target_horizon_c[i, i]
                    temperature_target_dict[self.district_id_dict[i]] = (
                        None if np.isnan(val) else float(val)
                    )

                co2_max_dict = {
                    self.district_id_dict[i]: float(co2_max_horizon_ppm[i, i])
                    for i in range(self.num_nodes)
                }

                is_occupied_dict = {
                    self.district_id_dict[i]: float(is_occupied_horizon[i, i])
                    for i in range(self.num_nodes)
                }

                forecast_data.append(
                    {
                        "time": future_time.isoformat(),
                        "q_w": q_dict,
                        "v_m3_s": v_dict,
                        "t_target_c": temperature_target_dict,
                        "co2_max_ppm": co2_max_dict,
                        "is_occupied": is_occupied_dict,
                    }
                )

                future_time += pd.Timedelta(seconds=dt_seconds)

            self.steps_since_last_recalc = 0

        if self.steps_since_last_recalc < len(self.cached_q_plan_w):  # type: ignore
            current_q_w = self.cached_q_plan_w[self.steps_since_last_recalc, :]  # type: ignore
            current_v_m3_s = self.cached_v_plan_m3_s[self.steps_since_last_recalc, :]  # type: ignore
            current_bess_kw = self.cached_bess_plan_kw[self.steps_since_last_recalc]  # type: ignore
        else:
            current_q_w = np.zeros(self.num_nodes)
            current_v_m3_s = np.zeros(self.num_nodes)
            current_bess_kw = 0.0

        self.steps_since_last_recalc += 1

        return current_q_w, current_v_m3_s, current_bess_kw, forecast_data


@njit(fastmath=True)
def mpc_cost_function(
    x_flat,
    control_steps,
    num_nodes,
    max_heating_power_w,
    max_cooling_power_w,
    max_vent_power_m3_s,
    bess_max_power_kw,
    bess_capacity_kwh,
    bess_efficiency,
    bess_min_soc,
    bess_max_soc,
    current_bess_soc,
    current_bess_kw,
    co2_generation_rates_m3_s,
    temperature_tolerance_c,
    dt_seconds,
    horizon_steps,
    current_temperature_c,
    current_co2_ppm,
    current_q_hvac_w,
    current_v_m3_s,
    contracted_power_kw,
    hrv_efficiency,
    temperature_target_horizon_c,
    co2_max_horizon_ppm,
    is_occupied_horizon,
    temperature_out_forecast_c,
    co2_out_forecast_ppm,
    q_env_forecast_w,
    elec_cost_forecast_eur_mwh,
    res_yield_forecast_kw,
    cop_heat_forecast,
    cop_cool_forecast,
    air_mixing_rate_m3_s,
    infiltration_rate_m3_s,
    volumes_m3,
    thermal_conductance_w_k,
    ext_air_conductance_w_k,
    ext_ground_conductance_w_k,
    ground_temperature_c,
    heat_capacity_j_k,
):
    idx_q_end = control_steps * num_nodes
    idx_v_end = idx_q_end + (control_steps * num_nodes)
    block_size = horizon_steps // control_steps

    temperature_sim_c = np.copy(current_temperature_c)
    co2_sim_ppm = np.copy(current_co2_ppm)

    soc_sim = float(current_bess_soc)
    prev_bess_kw = float(current_bess_kw)

    prev_q_hvac_w = np.copy(current_q_hvac_w)
    prev_v_hvac_m3_s = np.copy(current_v_m3_s)

    total_penalty = 0

    micro_steps = int(np.ceil(dt_seconds / 60))
    micro_dt_s = dt_seconds / micro_steps

    sum_air_mixing = np.sum(air_mixing_rate_m3_s, axis=1)
    sum_thermal_cond = np.sum(thermal_conductance_w_k, axis=1)

    for i in range(horizon_steps):
        c_idx = i // block_size

        q_hvac_w = np.zeros(num_nodes)
        v_vent_m3_s = np.zeros(num_nodes)

        v_perc_array = np.zeros(num_nodes)
        q_perc_array = np.zeros(num_nodes)

        for j in range(num_nodes):
            idx_q = (c_idx * num_nodes) + j
            idx_v = idx_q_end + (c_idx * num_nodes) + j

            q_perc = x_flat[idx_q]
            v_perc = x_flat[idx_v]

            q_perc_array[j] = q_perc
            v_perc_array[j] = v_perc

            if q_perc >= 0:
                q_hvac_w[j] = (q_perc / 100.0) * max_heating_power_w[j]
            else:
                q_hvac_w[j] = (q_perc / 100.0) * max_cooling_power_w[j]

            v_vent_m3_s[j] = (v_perc / 100.0) * max_vent_power_m3_s

        idx_bess = idx_v_end + c_idx
        bess_perc = x_flat[idx_bess]
        requested_bess_kw = (bess_perc / 100.0) * bess_max_power_kw

        dt_hours = dt_seconds / 3600.0
        actual_bess_kw = 0.0

        if requested_bess_kw > 0:
            available_capacity_kwh = (bess_max_soc - soc_sim) * bess_capacity_kwh
            max_charge_kw = (available_capacity_kwh / bess_efficiency) / dt_hours
            actual_bess_kw = min(requested_bess_kw, max_charge_kw)
            soc_sim += (actual_bess_kw * dt_hours * bess_efficiency) / bess_capacity_kwh
        elif requested_bess_kw < 0:
            available_energy_kwh = (soc_sim - bess_min_soc) * bess_capacity_kwh
            max_discharge_kw = -(available_energy_kwh * bess_efficiency / dt_hours)
            actual_bess_kw = max(requested_bess_kw, max_discharge_kw)
            soc_sim += (actual_bess_kw * dt_hours / bess_efficiency) / bess_capacity_kwh

        total_penalty += calculate_control_penalties(
            v_perc_array,
            q_perc_array,
            prev_v_hvac_m3_s,
            prev_q_hvac_w,
            max_vent_power_m3_s,
            max_heating_power_w,
            max_cooling_power_w,
            actual_bess_kw,
            prev_bess_kw,
            num_nodes,
        )

        prev_bess_kw = actual_bess_kw

        for j in range(num_nodes):
            prev_v_hvac_m3_s[j] = v_vent_m3_s[j]
            prev_q_hvac_w[j] = q_hvac_w[j]

        co2_gen_m3_s = co2_generation_rates_m3_s * is_occupied_horizon[i]

        for _ in range(micro_steps):
            co2_mixed = np.dot(air_mixing_rate_m3_s, co2_sim_ppm) - (
                sum_air_mixing * co2_sim_ppm
            )
            co2_infil = infiltration_rate_m3_s * (co2_out_forecast_ppm[i] - co2_sim_ppm)
            co2_vent = v_vent_m3_s * (co2_out_forecast_ppm[i] - co2_sim_ppm)

            total_co2_flow = co2_mixed + co2_infil + co2_vent
            co2_sim_ppm += (total_co2_flow / volumes_m3) * micro_dt_s
            co2_sim_ppm += (co2_gen_m3_s / volumes_m3) * 1000000 * micro_dt_s

        q_inter_w = np.dot(thermal_conductance_w_k, temperature_sim_c) - (
            sum_thermal_cond * temperature_sim_c
        )
        q_air_w = ext_air_conductance_w_k * (
            temperature_out_forecast_c[i] - temperature_sim_c
        )
        q_ground_w = ext_ground_conductance_w_k * (
            ground_temperature_c - temperature_sim_c
        )

        target_temperature = temperature_target_horizon_c[i]
        safe_target_temperature = np.where(
            np.isnan(target_temperature), 21.0, target_temperature
        )

        bypass_active = (temperature_out_forecast_c[i] < temperature_sim_c) & (
            temperature_sim_c > safe_target_temperature
        )
        effective_eff = np.where(bypass_active, 0.0, hrv_efficiency)

        q_vent_w = (
            v_vent_m3_s
            * 1200
            * (1 - effective_eff)
            * (temperature_out_forecast_c[i] - temperature_sim_c)
        )

        total_q_w = (
            q_inter_w + q_air_w + q_ground_w + q_vent_w + q_env_forecast_w[i] + q_hvac_w
        )

        temperature_sim_c += (total_q_w / heat_capacity_j_k) * dt_seconds

        total_penalty += calculate_comfort_penalties(
            temperature_sim_c,
            co2_sim_ppm,
            target_temperature,
            temperature_tolerance_c,
            co2_max_horizon_ppm,
            num_nodes,
            i,
        )

        total_penalty += calculate_financial_and_grid_penalties(
            q_hvac_w,
            v_vent_m3_s,
            actual_bess_kw,
            cop_heat_forecast[i],
            cop_cool_forecast[i],
            elec_cost_forecast_eur_mwh[i],
            res_yield_forecast_kw[i],
            contracted_power_kw,
        )

    return total_penalty


@njit(fastmath=True)
def calculate_control_penalties(
    v_perc_array,
    q_perc_array,
    prev_v_hvac_m3_s_array,
    prev_q_hvac_w_array,
    max_vent_power_m3_s,
    max_heating_power_w,
    max_cooling_power_w,
    actual_bess_kw,
    prev_bess_kw,
    num_nodes,
):
    VENTILATION_PENALTY_WEIGHT = 1.0
    VENTILATION_CHANGE_PENALTY_WEIGHT = 5.0
    Q_CHANGE_PENALTY_WEIGHT = 5.0
    BESS_CHANGE_PENALTY_WEIGHT = 0.0

    penalty = 0.0

    for i in range(num_nodes):
        v_perc = v_perc_array[i]
        q_perc = q_perc_array[i]

        # PENALTY: for ventilating (noise, filter wear)
        penalty += (v_perc**2) * VENTILATION_PENALTY_WEIGHT

        prev_v_perc = (prev_v_hvac_m3_s_array[i] / max_vent_power_m3_s) * 100.0

        # PENALTY: for ventilation changing (fan slew rate)
        delta_v_perc = v_perc - prev_v_perc
        penalty += (delta_v_perc**2) * VENTILATION_CHANGE_PENALTY_WEIGHT

        prev_q_w = prev_q_hvac_w_array[i]
        if prev_q_w >= 0.0:
            prev_q_perc = (
                (prev_q_w / max_heating_power_w[i]) * 100.0
                if max_heating_power_w[i] > 0
                else 0.0
            )
        else:
            prev_q_perc = (
                (prev_q_w / max_cooling_power_w[i]) * 100.0
                if max_cooling_power_w[i] > 0
                else 0.0
            )

        # PENALTY: for changing Q (compressor slew rate)
        delta_q_perc = q_perc - prev_q_perc
        penalty += (delta_q_perc**2) * Q_CHANGE_PENALTY_WEIGHT

    # PENALTY: for changing BESS power (battery wear)
    delta_bess_kw = actual_bess_kw - prev_bess_kw
    penalty += (delta_bess_kw**2) * BESS_CHANGE_PENALTY_WEIGHT

    return penalty


@njit(fastmath=True)
def calculate_comfort_penalties(
    temperature_sim_c,
    co2_sim_ppm,
    target_temperature,
    temperature_tolerance_c,
    co2_max_horizon_ppm,
    num_nodes,
    horizon_idx,
):
    TEMP_BELOW_PENALTY_WEIGHT = 100000.0
    TEMP_ABOVE_PENALTY_WEIGHT = 5000.0
    FREEZE_PENALTY_WEIGHT = 100000.0
    CO2_PENALTY_WEIGHT = 50.0

    penalty = 0.0

    for i in range(num_nodes):
        # PENALTY: for temperature outside the bounds
        if not np.isnan(target_temperature[i]):
            below_min = max(
                0.0,
                (target_temperature[i] - temperature_tolerance_c)
                - temperature_sim_c[i],
            )
            above_max = max(
                0.0,
                temperature_sim_c[i]
                - (target_temperature[i] + temperature_tolerance_c),
            )

            penalty += (below_min**2) * TEMP_BELOW_PENALTY_WEIGHT
            penalty += (above_max**2) * TEMP_ABOVE_PENALTY_WEIGHT

        # PENALTY: for freezing temperature (pipe burst risk)
        pipe_freeze_risk = max(0.0, 16.0 - temperature_sim_c[i])
        penalty += (pipe_freeze_risk**2) * FREEZE_PENALTY_WEIGHT

        # PENALTY: for exceeding CO2 level
        co2_violation = max(0.0, co2_sim_ppm[i] - co2_max_horizon_ppm[horizon_idx, i])
        penalty += (co2_violation**2) * CO2_PENALTY_WEIGHT

    return penalty


@njit(fastmath=True)
def calculate_financial_and_grid_penalties(
    q_hvac_w_array,
    v_vent_m3_s_array,
    bess_power_kw,
    cop_heat,
    cop_cool,
    elec_cost_eur_mwh,
    res_yield_kw,
    contracted_power_kw,
):
    COST_PENALTY_WEIGHT = 10000.0
    POWER_EXCESS_PENALTY_WEIGHT = 100000.0

    penalty = 0.0

    q_heating_w = np.maximum(0.0, q_hvac_w_array)
    q_cooling_w = np.maximum(0.0, -q_hvac_w_array)

    # TODO: improve that by considering actual power curves of heat pump and fan, instead of simple linear model
    v_power_w = v_vent_m3_s_array * 1000.0

    heat_elec_demand_kw = (np.sum(q_heating_w) / cop_heat) / 1000.0
    cool_elec_demand_kw = (np.sum(q_cooling_w) / cop_cool) / 1000.0
    vent_elec_demand_kw = np.sum(v_power_w) / 1000.0

    total_elec_demand_kw = (
        heat_elec_demand_kw + cool_elec_demand_kw + vent_elec_demand_kw
    )

    total_effective_demand_kw = max(0.0, total_elec_demand_kw + bess_power_kw)
    bess_export_kw = max(0.0, -(total_elec_demand_kw + bess_power_kw))

    if elec_cost_eur_mwh < 0:
        actual_pv_yield_kw = min(res_yield_kw, total_effective_demand_kw)
        grid_sell_kw = bess_export_kw
    else:
        actual_pv_yield_kw = res_yield_kw
        grid_sell_kw = (
            max(0.0, actual_pv_yield_kw - total_effective_demand_kw) + bess_export_kw
        )

    grid_buy_kw = max(0.0, total_effective_demand_kw - actual_pv_yield_kw)

    # PENALTY: for high costs
    step_financial_cost = (grid_buy_kw * elec_cost_eur_mwh) - (
        grid_sell_kw * elec_cost_eur_mwh
    )
    penalty += step_financial_cost * COST_PENALTY_WEIGHT

    # PENALTY: for exceeding max contracted power
    if grid_buy_kw > contracted_power_kw:
        excess_kw = grid_buy_kw - contracted_power_kw
        penalty += (excess_kw**2) * POWER_EXCESS_PENALTY_WEIGHT

    return penalty
