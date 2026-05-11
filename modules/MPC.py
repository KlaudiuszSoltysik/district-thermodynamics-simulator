import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shared.MongoDbController import MongoDbController


class MPC:
    def __init__(self, pv_farm, heat_pump, num_nodes, district_id_dict):
        self.horizon_hours = 6
        self.block_size = 12

        self.cached_t_plan = None
        self.cached_v_plan = None
        self.plan_step_index = 0

        self.mongodb = MongoDbController()

        self.pv_farm = pv_farm
        self.heat_pump = heat_pump
        self.num_nodes = num_nodes
        self.max_heat_pump_powers = heat_pump.max_heat_pump_powers
        self.min_heat_pump_powers = heat_pump.min_heat_pump_powers
        self.district_id_dict = district_id_dict

        self.target_24h = None
        self.min_24h = None
        self.max_24h = None
        self.is_enabled_24h = None
        self.set_temperatures_config()

    def set_temperatures_config(self):
        self.target_24h = np.full((self.num_nodes, 24), 21.0)
        tolerance_channel = np.full((self.num_nodes, 24), 0.5)
        self.is_enabled_24h = np.zeros((self.num_nodes, 24))

        configs = list(self.mongodb.db["apartments-config"].find({}))

        mongo_map = {}
        for apt in configs:
            b_id = apt["BuildingId"]
            a_id = apt["ApartmentId"]

            for room in apt["Rooms"]:
                r_id = room["_id"]
                flat_key = f"{b_id}:{a_id}:{r_id}"
                mongo_map[flat_key] = room["HvacControl"]

        for idx, full_id in self.district_id_dict.items():
            if full_id in mongo_map:
                ctrl = mongo_map[full_id]

                temps = ctrl["Temperatures"]
                if temps and len(temps) == 24:
                    self.target_24h[idx, :] = temps

                tolerance_channel[idx] = ctrl["Tolerance"]

                is_enabled = ctrl["IsEnabled"]
                if is_enabled and len(is_enabled) == 24:
                    self.is_enabled_24h[idx, :] = [
                        1.0 if s else 0.0 for s in is_enabled
                    ]

        self.min_24h = self.target_24h - tolerance_channel
        self.max_24h = self.target_24h + tolerance_channel

    def get_target_trajectories(self, current_time, dt, horizon_steps):
        T_min_horizon = np.zeros((horizon_steps, self.num_nodes))
        T_max_horizon = np.zeros((horizon_steps, self.num_nodes))

        future_time = current_time
        for k in range(horizon_steps):
            time_float = future_time.hour + (future_time.minute / 60.0)
            h0 = int(time_float) % 24
            h1 = (h0 + 1) % 24
            w = time_float - int(time_float)
            mu = (1.0 - np.cos(w * np.pi)) / 2.0

            base_min = self.min_24h[:, h0] * (1.0 - mu) + self.min_24h[:, h1] * mu
            base_max = self.max_24h[:, h0] * (1.0 - mu) + self.max_24h[:, h1] * mu

            is_active = self.is_enabled_24h[:, h0] > 0.5

            T_min_horizon[k, :] = np.where(is_active, base_min, 16.0)
            T_max_horizon[k, :] = np.where(is_active, base_max, 26.0)

            future_time += pd.Timedelta(seconds=dt)

        return T_min_horizon, T_max_horizon

    def cost_function(
        self,
        x_flat,
        current_T,
        current_co2,
        T_min_horizon,
        T_max_horizon,
        t_out_forecast,
        co2_out_forecast,
        q_env_forecast,
        is_enabled_forecast,
        thermal_solver,
        co2_solver,
        dt,
        horizon_steps,
        control_steps,
        elec_cost_forecast,
        gas_cost_forecast,
        res_yield_forecast,
        cop_heat_forecast,
        cop_cool_forecast,
    ):
        half_idx = control_steps * self.num_nodes

        Q_percent_blocked = x_flat[:half_idx].reshape((control_steps, self.num_nodes))
        V_percent_blocked = x_flat[half_idx:].reshape((control_steps, self.num_nodes))

        Q_watts_blocked = np.where(
            Q_percent_blocked >= 0,
            (Q_percent_blocked / 100.0) * self.max_heat_pump_powers,
            (Q_percent_blocked / 100.0) * self.min_heat_pump_powers,
        )
        V_m3s_blocked = (V_percent_blocked / 100.0) * 0.05

        Q_hvac_matrix = np.repeat(Q_watts_blocked, self.block_size, axis=0)
        V_vent_matrix = np.repeat(V_m3s_blocked, self.block_size, axis=0)

        T_sim = np.array(current_T, dtype=float)
        co2_sim = np.array(current_co2, dtype=float)
        total_penalty = 0.0

        G = thermal_solver.G_temp
        C = thermal_solver.C
        G_ext_air = thermal_solver.G_ext_air
        G_ext_ground = thermal_solver.G_ext_ground
        T_ground = thermal_solver.T_ground

        micro_steps = int(np.ceil(dt / 60))
        micro_dt = dt / micro_steps

        for k in range(horizon_steps):
            Q_hvac = Q_hvac_matrix[k]
            V_vent = V_vent_matrix[k]

            co2_generation_m3_s = (0.015 / 3600.0) * is_enabled_forecast[k]

            for _ in range(micro_steps):
                co2_mixed = np.dot(co2_solver.G, co2_sim) - (
                    np.sum(co2_solver.G, axis=1) * co2_sim
                )
                co2_infil = co2_solver.G_ext_air_mix * (co2_out_forecast[k] - co2_sim)
                co2_vent = V_vent * (co2_out_forecast[k] - co2_sim)

                total_co2 = co2_mixed + co2_infil + co2_vent
                co2_sim += (total_co2 / co2_solver.V) * micro_dt
                co2_sim += (co2_generation_m3_s / co2_solver.V) * 1_000_000.0 * micro_dt

            co2_sim = np.maximum(co2_sim, 400.0)

            Q_inter = np.dot(G, T_sim) - (np.sum(G, axis=1) * T_sim)
            Q_air = G_ext_air * (t_out_forecast[k] - T_sim)
            Q_ground = G_ext_ground * (T_ground - T_sim)
            Q_vent = V_vent * 1200.0 * (1.0 - 0.8) * (t_out_forecast[k] - T_sim)

            total_Q = Q_inter + Q_air + Q_ground + Q_vent + q_env_forecast[k] + Q_hvac
            T_sim += (total_Q / C) * dt

            # penalty for temperatures outside band
            below_min = np.maximum(0, T_min_horizon[k] - T_sim)
            above_max = np.maximum(0, T_sim - T_max_horizon[k])
            total_penalty += (
                np.sum(below_min) * 10000.0 + np.sum(below_min**2) * 50000.0
            )
            total_penalty += (
                np.sum(above_max) * 10000.0 + np.sum(above_max**2) * 50000.0
            )

            # penalty for freezing floor
            floor_freezing_penalty = np.maximum(0, 19.0 - T_sim)
            total_penalty += np.sum(floor_freezing_penalty) * 100000.0

            # penalty for exceeding co2 level
            co2_suffocation = np.maximum(0, co2_sim - 1000.0)
            total_penalty += (
                np.sum(co2_suffocation) * 1000.0 + np.sum(co2_suffocation**2) * 5000.0
            )

            # penalty for energy expenses
            Q_heating = np.maximum(0, Q_hvac)
            Q_cooling = np.maximum(0, -Q_hvac)

            V_power_watts = V_vent * 1000.0

            heat_elec_demand = np.sum(Q_heating) / cop_heat_forecast[k]
            cool_elec_demand = np.sum(Q_cooling) / cop_cool_forecast[k]
            vent_elec_demand = np.sum(V_power_watts)

            total_elec_demand_kw = (
                heat_elec_demand + cool_elec_demand + vent_elec_demand
            ) / 1000.0

            grid_buy_kw = np.maximum(0, total_elec_demand_kw - res_yield_forecast[k])

            step_cost = grid_buy_kw * elec_cost_forecast[k]

            # Jeśli w przyszłości dodasz "Q_gas" jako osobną zmienną, tutaj dodasz:
            # step_cost += (np.sum(Q_gas)/1000.0 / 0.95) * gas_cost_for[k]

            total_penalty += step_cost * 50.0

            # penalty for using hvac
            total_penalty += np.sum(np.abs(Q_hvac)) * 0.01
            total_penalty += np.sum(V_vent) * 100.0

        # penalty for low stability
        delta_q_frac = np.diff(Q_percent_blocked, axis=0) / 100.0
        delta_v_frac = np.diff(V_percent_blocked, axis=0) / 100.0
        total_penalty += np.sum(delta_q_frac**2) * 5000.0
        total_penalty += np.sum(delta_v_frac**2) * 5000.0

        # penalty for exceeding maximum power change
        illegal_jumps = np.maximum(0, np.abs(delta_q_frac) - 0.15)
        total_penalty += np.sum(illegal_jumps) * 10000000.0

        return total_penalty

    def step(
        self,
        current_time,
        dt,
        weather_solver,
        thermal_solver,
        co2_solver,
        weather_service,
        energy_service,
    ):
        horizon_steps = int((self.horizon_hours * 3600) / dt)
        needs_recalc = False

        if self.cached_t_plan is None:
            needs_recalc = True
        elif self.plan_step_index >= 12:
            needs_recalc = True

        if needs_recalc:
            t_out_forecast = np.zeros(horizon_steps)
            co2_out_forecast = np.zeros(horizon_steps)
            q_env_forecast = np.zeros((horizon_steps, self.num_nodes))
            is_enabled_forecast = np.zeros((horizon_steps, self.num_nodes))

            elec_cost_forecast = np.zeros(horizon_steps)
            gas_cost_forecast = np.zeros(horizon_steps)
            res_yield_forecast = np.zeros(horizon_steps)
            cop_heat_forecast = np.zeros(horizon_steps)
            cop_cool_forecast = np.zeros(horizon_steps)

            future_time = current_time
            T_frozen_for_prediction = np.copy(thermal_solver.T)

            for k in range(horizon_steps):
                w = weather_service.get_weather(future_time)
                t_out_forecast[k] = w["temperature"]
                co2_out_forecast[k] = w["co2"]

                q_env = weather_solver.calculate_environmental_gains(
                    w["sun_radiation"],
                    w["sun_altitude"],
                    w["sun_azimuth"],
                    w["wind_speed"],
                    w["wind_direction"],
                    w["temperature"],
                    T_frozen_for_prediction,
                )
                q_env_forecast[k, :] = q_env

                time_float = future_time.hour + (future_time.minute / 60.0)
                current_h = int(time_float) % 24
                is_enabled_forecast[k, :] = co2_solver.is_enabled_mask[:, current_h]

                costs = energy_service.get_effective_costs(
                    future_time, self.pv_farm, self.heat_pump, w
                )

                elec_cost_forecast[k] = costs["electricity_price"]
                gas_cost_forecast[k] = costs["gas_price"]
                res_yield_forecast[k] = costs["pv_farm_yield"]
                cop_heat_forecast[k] = costs["cop_heating"]
                cop_cool_forecast[k] = costs["cop_cooling"]

                future_time += pd.Timedelta(seconds=dt)

            T_min_horizon, T_max_horizon = self.get_target_trajectories(
                current_time, dt, horizon_steps
            )

            control_steps = horizon_steps // self.block_size

            bounds_q = [
                (-100.0, 100.0)
                for _ in range(control_steps)
                for i in range(self.num_nodes)
            ]
            bounds_v = [
                (0.0, 100.0)
                for _ in range(control_steps)
                for i in range(self.num_nodes)
            ]
            bounds = bounds_q + bounds_v

            initial_guess = np.zeros(control_steps * self.num_nodes * 2)

            res = minimize(
                self.cost_function,
                initial_guess,
                args=(
                    thermal_solver.T,
                    co2_solver.co2,
                    T_min_horizon,
                    T_max_horizon,
                    t_out_forecast,
                    co2_out_forecast,
                    q_env_forecast,
                    is_enabled_forecast,
                    thermal_solver,
                    co2_solver,
                    dt,
                    horizon_steps,
                    control_steps,
                    elec_cost_forecast,
                    gas_cost_forecast,
                    res_yield_forecast,
                    cop_heat_forecast,
                    cop_cool_forecast,
                ),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 1,  # 10
                    "ftol": 1e-3,
                    "eps": 2.0,
                    "disp": False,
                },
            )

            half_idx = control_steps * self.num_nodes
            optimal_q_percent = res.x[:half_idx].reshape(
                (control_steps, self.num_nodes)
            )
            optimal_v_percent = res.x[half_idx:].reshape(
                (control_steps, self.num_nodes)
            )

            optimal_q_watts = np.where(
                optimal_q_percent >= 0,
                (optimal_q_percent / 100.0) * self.max_heat_pump_powers,
                (optimal_q_percent / 100.0) * self.min_heat_pump_powers,
            )
            optimal_v_m3s = (optimal_v_percent / 100.0) * 0.05

            self.cached_t_plan = np.repeat(optimal_q_watts, self.block_size, axis=0)
            self.cached_v_plan = np.repeat(optimal_v_m3s, self.block_size, axis=0)
            self.plan_step_index = 0

        if self.plan_step_index < len(self.cached_t_plan):
            current_optimal_q = self.cached_t_plan[self.plan_step_index, :]
            current_optimal_v = self.cached_v_plan[self.plan_step_index, :]
        else:
            current_optimal_q = np.zeros(self.num_nodes)
            current_optimal_v = np.zeros(self.num_nodes)

        self.plan_step_index += 1

        return current_optimal_q, current_optimal_v
