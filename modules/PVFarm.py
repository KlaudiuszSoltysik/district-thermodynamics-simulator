import numpy as np


# TODO: Enhance physics
class PVFarm:
    STC_RADIATION_W_M2 = 1000

    def __init__(self, max_power_kw, efficiency):
        self.max_power_kw = max_power_kw
        self.efficiency = efficiency

    def get_power_prognosis(self, weather):
        base_yield_kw = (
            self.max_power_kw
            * self.efficiency
            * (weather.sun_radiation_w_m2 / self.STC_RADIATION_W_M2)
        )

        return float(np.clip(base_yield_kw, 0, self.max_power_kw))
