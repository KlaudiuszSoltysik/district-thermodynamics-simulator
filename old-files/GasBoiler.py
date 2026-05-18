class GasBoiler:
    def __init__(self, max_power=50, min_efficiency=0.9, max_efficiency=0.98):
        self.max_power = max_power
        self.min_efficiency = min_efficiency
        self.max_efficiency = max_efficiency

    def get_efficiency(self, current_power):
        if current_power <= 0:
            return self.max_efficiency

        part_load = current_power / self.max_power

        part_load = min(1, max(0, part_load))

        efficiency = self.min_efficiency + (1 - part_load) * (
            self.max_efficiency - self.min_efficiency
        )

        return efficiency
