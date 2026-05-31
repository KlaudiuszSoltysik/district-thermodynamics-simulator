# TODO: Improve BESS model with C-rate, degradation, and more detailed efficiency curves.
class BESS:
    def __init__(self, capacity_kwh, max_power_kw, efficiency, min_soc, max_soc):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.efficiency = efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc

        self.current_soc = self.min_soc

    def step(self, dt_seconds, requested_power_kw):
        requested_power_kw = max(
            -self.max_power_kw, min(self.max_power_kw, requested_power_kw)
        )

        dt_hours = dt_seconds / 3600.0
        actual_power_kw = 0.0

        if requested_power_kw > 0:
            available_capacity_kwh = (
                self.max_soc - self.current_soc
            ) * self.capacity_kwh
            max_charge_kwh = available_capacity_kwh / self.efficiency
            max_charge_kw = max_charge_kwh / dt_hours

            actual_power_kw = min(requested_power_kw, max_charge_kw)
            self.current_soc += (
                actual_power_kw * dt_hours * self.efficiency
            ) / self.capacity_kwh

        elif requested_power_kw < 0:
            available_energy_kwh = (self.current_soc - self.min_soc) * self.capacity_kwh
            max_discharge_kwh = available_energy_kwh * self.efficiency
            max_discharge_kw = -(max_discharge_kwh / dt_hours)

            actual_power_kw = max(requested_power_kw, max_discharge_kw)
            self.current_soc += (
                actual_power_kw * dt_hours / self.efficiency
            ) / self.capacity_kwh

        return actual_power_kw, self.current_soc
