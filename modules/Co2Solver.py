import numpy as np
from shared.MongoDbController import MongoDbController


class Co2Solver:
    def __init__(self, G, V, G_ext_air_mix, num_nodes, district_id_dict):
        self.G = G
        self.V = V
        self.G_ext_air_mix = G_ext_air_mix
        self.num_nodes = num_nodes
        self.district_id_dict = district_id_dict

        self.co2 = np.full(len(V), 750.0)

        self.mongodb = MongoDbController()

        self.is_enabled_mask = None

        self.set_on_hours()

    def set_on_hours(self):
        self.is_enabled_mask = np.zeros((self.num_nodes, 24))

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
                is_enabled = mongo_map[full_id]["IsEnabled"]

                if is_enabled and len(is_enabled) == 24:
                    self.is_enabled_mask[idx, :] = [
                        1.0 if s else 0.0 for s in is_enabled
                    ]

    def step(self, current_time, dt, outside_co2, v_hvac):
        time_float = current_time.hour + (current_time.minute / 60.0)
        current_h = int(time_float) % 24

        current_mask = self.is_enabled_mask[:, current_h]

        steps = int(np.ceil(dt / 60))
        micro_dt = dt / steps

        co2_generation_m3_s = (0.015 / 3600.0) * current_mask

        for _ in range(steps):
            co2_mixed = np.dot(self.G, self.co2) - (np.sum(self.G, axis=1) * self.co2)
            co2_infiltrated = self.G_ext_air_mix * (outside_co2 - self.co2)
            co2_vented = v_hvac * (outside_co2 - self.co2)

            total_co2_flow_per_hour = co2_mixed + co2_infiltrated + co2_vented

            total_co2_flow_per_second = total_co2_flow_per_hour / 3600.0

            self.co2 += (total_co2_flow_per_second / self.V) * micro_dt
            self.co2 += (co2_generation_m3_s / self.V) * 1000000.0 * micro_dt

        self.co2 = np.maximum(self.co2, 400.0)

        return np.round(self.co2).astype(int)
