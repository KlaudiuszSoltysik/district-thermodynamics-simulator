from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib


@dataclass
class WeatherData:
    temperature_c: float
    wind_speed_m_s: float
    wind_direction_deg: float
    sun_radiation_w_m2: float
    sun_altitude_deg: float
    sun_azimuth_deg: float
    co2_ppm: int


class WeatherService:
    def __init__(self, weather_path, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

        df = pd.read_csv(weather_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        rad = np.radians(df["wind_direction"])
        df["wind_u"] = np.sin(rad)
        df["wind_v"] = np.cos(rad)

        self.weather_history = df.set_index("timestamp").sort_index()

    def get_weather(self, current_time):
        idx_after = self.weather_history.index.searchsorted(current_time)

        if idx_after == 0:
            idx_after = 1
        if idx_after >= len(self.weather_history):
            idx_after = len(self.weather_history) - 1

        t1 = self.weather_history.index[idx_after - 1]  # type: ignore
        t2 = self.weather_history.index[idx_after]  # type: ignore

        weight2 = (current_time - t1).total_seconds() / (t2 - t1).total_seconds()
        weight1 = 1 - weight2

        interp_row = (self.weather_history.loc[t1] * weight1) + (
            self.weather_history.loc[t2] * weight2
        )

        wind_dir_rad = np.arctan2(interp_row["wind_u"], interp_row["wind_v"])
        wind_direction_deg = (np.degrees(wind_dir_rad) + 360) % 360

        solar_pos = pvlib.solarposition.get_solarposition(
            time=pd.DatetimeIndex([current_time]),
            latitude=self.latitude,
            longitude=self.longitude,
        )

        return WeatherData(
            temperature_c=float(interp_row["temperature"]),
            wind_speed_m_s=float(interp_row["wind_speed"]),
            wind_direction_deg=float(wind_direction_deg),
            sun_radiation_w_m2=float(interp_row.get("sun_radiation", 0)),
            sun_altitude_deg=float(solar_pos["apparent_elevation"].iloc[0]),  # type: ignore
            sun_azimuth_deg=float(solar_pos["azimuth"].iloc[0]),  # type: ignore
            co2_ppm=int(round(interp_row["co2"])),
        )
