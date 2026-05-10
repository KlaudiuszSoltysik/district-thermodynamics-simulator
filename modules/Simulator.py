from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import yaml

from modules.DistrictConfigParser import DistrictModelParser
from modules.WeatherService import WeatherService


@dataclass
class SimulationStep:
    time: str
    out_temperature_c: float
    out_wind_speed_m_s: float
    out_wind_direction_deg: float
    out_sun_radiation_w_m2: float
    out_sun_altitude_deg: float
    out_sun_azimuth_deg: float
    out_co2_ppm: int


class Simulator:
    START_TIMESTAMP = pd.Timestamp("2024-12-31 23:00+00:00")

    def __init__(self, district_config_path, weather_path, prices_path, logger):
        self.logger = logger
        self.logger.info("Initializing DistrictSimulation instance...")

        try:
            with open(district_config_path, "r", encoding="utf-8") as f:
                district_data = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load YAML config: {e}.")
            raise

        parser = DistrictModelParser(district_data)
        parser.parse()

        self.metadata = parser.metadata

        self.current_time = self.START_TIMESTAMP
        self.end_timestamp = self.START_TIMESTAMP + timedelta(days=365)

        self.weather_service = WeatherService(
            weather_path, self.metadata["latitude"], self.metadata["longitude"]
        )

        #         self.num_nodes = parser.N

        #
        #         self.index_to_id = {v: k for k, v in parser.nodes.items()}
        #
        #         self.thermal_solver = ThermalSolver(
        #             parser.G_temp,
        #             parser.C,
        #             parser.G_ext_air,
        #             parser.G_ext_ground,
        #             self.metadata["ground_temperature"],
        #             parser.A,
        #         )
        #
        #         self.co2_solver = Co2Solver(
        #             parser.G_air,
        #             parser.V,
        #             parser.G_ext_air_mix,
        #             self.num_nodes,
        #             self.index_to_id,
        #         )
        #
        #         self.weather_solver = WeatherSolver(
        #             parser.external_connections, parser.standards, self.num_nodes
        #         )
        #

        #         self.energy_service = EnergyService(prices_path)
        #
        #         self.pv_farm = PVFarm()
        #         self.heat_pump = HeatPump(
        #             parser.max_heat_pump_powers, parser.min_heat_pump_powers
        #         )
        #         self.gas_boiler = GasBoiler()
        #
        #         self.mpc = MPC(
        #             self.pv_farm,
        #             self.heat_pump,
        #             self.gas_boiler,
        #             self.num_nodes,
        #             parser.max_heat_pump_powers,
        #             self.index_to_id,
        #         )
        #
        #         self.metering_service = MeteringService(
        #             parser.A, self.num_nodes, self.index_to_id
        #         )

        self.logger.info("Simulation configured.")

    def run_step(self, dt):
        if self.current_time >= self.end_timestamp:
            self.logger.info("Simulation reached the end timestamp.")
            return

        weather = self.weather_service.get_weather(self.current_time)

        # energy_costs = self.energy_service.get_effective_costs(
        #     self.current_time, self.pv_farm, self.heat_pump, weather, noise_sigma
        # )
        #
        #         q_env = self.weather_solver.calculate_environmental_gains(
        #             weather["sun_radiation"],
        #             weather["sun_altitude"],
        #             weather["sun_azimuth"],
        #             weather["wind_speed"],
        #             weather["wind_direction"],
        #             weather["temperature"],
        #             self.thermal_solver.T,
        #         )
        #
        #         q_hvac, v_hvac = self.mpc.step(
        #             self.current_time,
        #             dt,
        #             self.thermal_solver,
        #             self.co2_solver,
        #             self.weather_service,
        #             self.weather_solver,
        #             self.energy_service,
        #             noise_sigma,
        #         )
        #
        #         q_total = q_env + q_hvac
        #
        #         temperatures_array = self.thermal_solver.step(
        #             dt, weather["temperature"], q_total, v_hvac, noise_sigma
        #         )
        #
        #         co2_array = self.co2_solver.step(
        #             self.current_time, dt, weather["co2"], v_hvac, noise_sigma
        #         )
        #
        #         self.metering_service.update_meters(
        #             self.current_time, dt, energy_costs, q_hvac, v_hvac
        #         )
        #
        #         energy_clean = {k: round(v, 2) for k, v in energy_costs.items()}
        #
        #         room_temps = {
        #             self.index_to_id[i]: round(float(temperatures_array[i]), 2)
        #             for i in range(self.num_nodes)
        #         }
        #         room_co2 = {
        #             self.index_to_id[i]: int(co2_array[i]) for i in range(self.num_nodes)
        #         }
        #         room_hvac_q = {
        #             self.index_to_id[i]: round(float(q_hvac[i]), 2)
        #             for i in range(self.num_nodes)
        #         }
        #
        #         denominators = np.where(
        #             q_hvac >= 0, self.mpc.max_heat_pump_powers, self.mpc.min_heat_pump_powers
        #         )
        #         q_percentage = (q_hvac / denominators) * 100.0
        #         room_heatings = {
        #             self.index_to_id[i]: round(float(q_percentage[i]), 2)
        #             for i in range(self.num_nodes)
        #         }
        #
        #         room_hvac_v = {
        #             self.index_to_id[i]: round(float(v_hvac[i] * 3600.0), 2)
        #             for i in range(self.num_nodes)
        #         }

        self.simulation_time = self.current_time

        self.current_time += timedelta(seconds=dt)

        return SimulationStep(
            time=self.current_time.isoformat(),
            out_temperature_c=weather.temperature_c,
            out_wind_speed_m_s=weather.wind_speed_m_s,
            out_wind_direction_deg=weather.wind_direction_deg,
            out_sun_radiation_w_m2=weather.sun_radiation_w_m2,
            out_sun_altitude_deg=weather.sun_altitude_deg,
            out_sun_azimuth_deg=weather.sun_azimuth_deg,
            out_co2_ppm=weather.co2_ppm,
        )
