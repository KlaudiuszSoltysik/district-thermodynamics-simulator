# TODO: Enhance physics
class HeatPump:
    def __init__(
        self,
        max_heating_power_w,
        max_cooling_power_w,
        base_cop=3,
        temp_modifier=0.1,
        min_cop=1,
        eer=3,
    ):
        self.max_heating_power_w = max_heating_power_w
        self.max_cooling_power_w = max_cooling_power_w
        self.base_cop = base_cop
        self.temp_modifier = temp_modifier
        self.min_cop = min_cop
        self.eer = eer

    def get_cop(self, weather):
        t_out_c = weather.temperature_c

        cop_heating = max(self.min_cop, self.base_cop + (self.temp_modifier * t_out_c))

        cop_cooling = self.eer

        return float(cop_heating), float(cop_cooling)
