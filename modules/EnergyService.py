from dataclasses import dataclass

import pandas as pd


@dataclass
class EnergyState:
    electricity_price_eur_per_mwh: float
    gas_price_eur_per_mwh: float
    pv_yield_kw: float
    cop_heating: float
    cop_cooling: float


class EnergyService:
    def __init__(self, prices_path):
        df = pd.read_csv(prices_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.prices_history = df.set_index("timestamp").sort_index()

    def get_effective_costs(
        self, current_time, pv_farm, heat_pump, weather
    ):
        current_hour = current_time.floor("h")

        idx_after = self.prices_history.index.searchsorted(current_hour)

        if idx_after >= len(self.prices_history):
            idx_after = len(self.prices_history) - 1

        cop_heating, cop_cooling = heat_pump.get_cop(weather)
        pv_yield_kw = pv_farm.get_power_prognosis(weather)

        electricity_price_eur_mwh = float(
            self.prices_history.iloc[idx_after]["electricity_price"]
        )
        gas_price_eur_mwh = float(self.prices_history.iloc[idx_after]["gas_price"])

        return EnergyState(
            electricity_price_eur_per_mwh=electricity_price_eur_mwh,
            gas_price_eur_per_mwh=gas_price_eur_mwh,
            pv_yield_kw=pv_yield_kw,
            cop_heating=cop_heating,
            cop_cooling=cop_cooling,
        )
