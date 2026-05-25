from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from modules.MeteringService import MeteringService
from modules.Co2Solver import Co2Solver
from modules.DistrictConfigParser import DistrictModelParser
from modules.EnergyService import EnergyService
from modules.HeatPump import HeatPump
from modules.MPC import MPC
from modules.PVFarm import PVFarm
from modules.ThermalSolver import ThermalSolver
from modules.WeatherService import WeatherService
from modules.WeatherSolver import WeatherSolver


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
    sys_electricity_price: float
    sys_gas_price: float
    sys_pv_yield_kw: float
    sys_cop_heating: float
    sys_cop_cooling: float
    room_temperatures_c: Dict[str, float]
    room_co2_ppm: Dict[str, int]
    room_q_hvac_w: Dict[str, float]
    room_q_hvac_perc: Dict[str, float]
    room_v_hvac_m3_s: Dict[str, float]
    meter_readings: Dict[str, Dict[str, Any]]
    mpc_forecast: Optional[List[Dict[str, Any]]] = None


class Simulator:
    START_TIMESTAMP = pd.Timestamp("2024-12-31 23:00+00:00")

    def __init__(
        self,
        district_config_path,
        weather_path,
        prices_path,
        dweller_schedule_patch,
        logger,
    ):
        self.logger = logger
        self.logger.info("Initializing DistrictSimulation instance...")

        with open(district_config_path, "r", encoding="utf-8") as f:
            district_data = yaml.safe_load(f)

        with open(dweller_schedule_patch, "r", encoding="utf-8") as f:
            dweller_schedule = yaml.safe_load(f)

        parser = DistrictModelParser(district_data)
        parser.parse()

        metadata = parser.metadata
        self.num_nodes = parser.num_rooms
        self.index_to_id = {v: k for k, v in parser.room_indices.items()}

        self.current_time = self.START_TIMESTAMP
        self.end_timestamp = self.START_TIMESTAMP + timedelta(days=365)

        self.weather_service = WeatherService(
            weather_path, metadata["latitude"], metadata["longitude"]
        )

        self.energy_service = EnergyService(prices_path)

        self.pv_farm = PVFarm(metadata["pv_max_power_kw"], metadata["pv_efficiency"])

        self.heat_pump = HeatPump(
            parser.max_heating_power_w, parser.min_heating_power_w
        )

        self.weather_solver = WeatherSolver(
            parser.external_connections, parser.standards, self.num_nodes
        )

        self.mpc = MPC(
            self.pv_farm,
            self.heat_pump,
            self.num_nodes,
            self.index_to_id,
            dweller_schedule,
        )

        self.thermal_solver = ThermalSolver(
            parser.thermal_conductance_w_k,
            parser.heat_capacity_j_k,
            parser.ext_air_conductance_w_k,
            parser.ext_ground_conductance_w_k,
            metadata["ground_temperature"],
            parser.areas_m2,
            self.num_nodes,
            self.index_to_id,
            dweller_schedule,
        )

        self.co2_solver = Co2Solver(
            parser.air_mixing_rate_m3_s,
            parser.volumes_m3,
            parser.infiltration_rate_m3_s,
            self.num_nodes,
            self.index_to_id,
            dweller_schedule,
        )

        self.metering_service = MeteringService(self.num_nodes, self.index_to_id)

        # self.gas_boiler = GasBoiler()

        self.logger.info("Simulation configured.")

    def run_step(self, dt):
        if self.current_time >= self.end_timestamp:
            self.logger.info("Simulation reached the end timestamp.")
            return

        weather = self.weather_service.get_weather(self.current_time)

        energy_costs = self.energy_service.get_effective_costs(
            self.current_time, self.pv_farm, self.heat_pump, weather
        )

        q_env_w = self.weather_solver.calculate_environmental_gains(
            weather.sun_radiation_w_m2,
            weather.sun_altitude_deg,
            weather.sun_azimuth_deg,
            weather.wind_speed_m_s,
            weather.wind_direction_deg,
            weather.temperature_c,
            self.thermal_solver.temperatures_c,
        )

        q_hvac_w, v_hvac_m3_s, forecast_data = self.mpc.step(
            self.current_time,
            dt,
            self.weather_solver,
            self.thermal_solver,
            self.co2_solver,
            self.weather_service,
            self.energy_service,
        )

        q_total_w = q_env_w + q_hvac_w

        self.metering_service.update_meters(dt, energy_costs, q_hvac_w, v_hvac_m3_s)
        meter_readings = self.metering_service.get_meter_readings()

        self.simulation_time = self.current_time

        temperatures_array_c = self.thermal_solver.step(
            self.current_time, dt, weather.temperature_c, q_total_w, v_hvac_m3_s
        )

        co2_array_ppm = self.co2_solver.step(
            self.current_time, dt, weather.co2_ppm, v_hvac_m3_s
        )

        room_temps = {
            self.index_to_id[i]: round(float(temperatures_array_c[i]), 2)
            for i in range(self.num_nodes)
        }

        room_co2 = {
            self.index_to_id[i]: int(co2_array_ppm[i]) for i in range(self.num_nodes)
        }

        room_q_w = {
            self.index_to_id[i]: float(q_hvac_w[i]) for i in range(self.num_nodes)
        }

        denominators = np.where(
            q_hvac_w >= 0, self.mpc.max_heating_power_w, self.mpc.max_cooling_power_w
        )
        q_hvac_perc = (q_hvac_w / denominators) * 100
        room_q_perc = {
            self.index_to_id[i]: float(q_hvac_perc[i]) for i in range(self.num_nodes)
        }

        room_v_m3_s = {
            self.index_to_id[i]: float(v_hvac_m3_s[i]) for i in range(self.num_nodes)
        }

        self.current_time += timedelta(seconds=dt)

        return SimulationStep(
            time=self.simulation_time.isoformat(),
            out_temperature_c=weather.temperature_c,
            out_wind_speed_m_s=weather.wind_speed_m_s,
            out_wind_direction_deg=weather.wind_direction_deg,
            out_sun_radiation_w_m2=weather.sun_radiation_w_m2,
            out_sun_altitude_deg=weather.sun_altitude_deg,
            out_sun_azimuth_deg=weather.sun_azimuth_deg,
            out_co2_ppm=weather.co2_ppm,
            sys_electricity_price=energy_costs.electricity_price_eur_per_mwh,
            sys_gas_price=energy_costs.gas_price_eur_per_mwh,
            sys_pv_yield_kw=energy_costs.pv_yield_kw,
            sys_cop_heating=energy_costs.cop_heating,
            sys_cop_cooling=energy_costs.cop_cooling,
            room_temperatures_c=room_temps,
            room_co2_ppm=room_co2,
            room_q_hvac_w=room_q_w,
            room_q_hvac_perc=room_q_perc,
            room_v_hvac_m3_s=room_v_m3_s,
            meter_readings=meter_readings,
            mpc_forecast=forecast_data,
        )
