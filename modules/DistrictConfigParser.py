import numpy as np


class DistrictModelParser:
    RHO_CP_AIR_J_M3K = 1200

    def __init__(self, district_data):
        self.raw_data = district_data
        self.metadata = self.raw_data.get("metadata", {})

        self.room_indices = {}
        self.room_info = []
        self.external_connections = []

        self._build_node_index()
        self.num_rooms = len(self.room_indices)

        self.thermal_conductance_w_k = np.zeros((self.num_rooms, self.num_rooms))
        self.ext_air_conductance_w_k = np.zeros(self.num_rooms)
        self.ext_ground_conductance_w_k = np.zeros(self.num_rooms)
        self.heat_capacity_j_k = np.zeros(self.num_rooms)

        self.air_mixing_rate_m3_s = np.zeros((self.num_rooms, self.num_rooms))
        self.infiltration_rate_m3_s = np.zeros(self.num_rooms)

        self.volumes_m3 = np.zeros(self.num_rooms)
        self.areas_m2 = np.zeros(self.num_rooms)

        self.max_heating_power_w = np.zeros(self.num_rooms)
        self.min_heating_power_w = np.zeros(self.num_rooms)

        self.standards = {
            b["id"]: b.get("standards", {}) for b in self.raw_data.get("buildings", [])
        }

    def _build_node_index(self):
        idx = 0
        for building in self.raw_data.get("buildings", []):
            for apartment in building.get("apartments", []):
                for room in apartment.get("rooms", []):
                    room_id = f"{building['id']}:{apartment['id']}:{room['id']}"

                    self.room_indices[room_id] = idx
                    self.room_info.append(
                        {"room": room, "standards": building.get("standards", {})}
                    )

                    idx += 1

    def parse(self):
        for i, data in enumerate(self.room_info):
            room = data["room"]
            standards = data["standards"]

            air_capacity_j_k = room["volume"] * self.RHO_CP_AIR_J_M3K
            capacity_key = room["heat_capacity_per_m2"]
            structure_capacity_j_k = (
                room["area"] * standards[capacity_key]["heat_capacity_per_m2"]
            )

            self.areas_m2[i] = room["area"]
            self.volumes_m3[i] = room["volume"]

            self.heat_capacity_j_k[i] = air_capacity_j_k + structure_capacity_j_k

            self.max_heating_power_w[i] = (
                room["area"] * standards["heating_power_per_m2"]
            )
            self.min_heating_power_w[i] = (
                room["area"] * standards["cooling_power_per_m2"]
            )

        for building in self.raw_data.get("buildings", []):
            b_id = building["id"]
            b_standards = building.get("standards", {})

            for connection in building.get("internal_connections", []):
                self._apply_internal_connection(connection, b_standards, b_id)

            for connection in building.get("external_connections", []):
                self._apply_external_connection(connection, b_standards, b_id)

    def _apply_internal_connection(self, connection, standards, building_id):
        from_str = connection["from"]
        to_str = connection["to"]

        idx_a = self.room_indices[f"{building_id}:{from_str}"]
        idx_b = self.room_indices[f"{building_id}:{to_str}"]

        code = standards[connection["thermal_code"]]

        ua_w_k = connection["area"] * code["u_value"]

        self.thermal_conductance_w_k[idx_a, idx_b] += ua_w_k
        self.thermal_conductance_w_k[idx_b, idx_a] += ua_w_k

        wall_capacity_j_k = connection["area"] * code["heat_capacity_per_m2"]
        self.heat_capacity_j_k[idx_a] += wall_capacity_j_k * 0.5
        self.heat_capacity_j_k[idx_b] += wall_capacity_j_k * 0.5

        apt_a = from_str.split(":")[0]
        apt_b = to_str.split(":")[0]

        if apt_a == apt_b:
            mixing_rate = standards.get("air_mixing_rate_m3_s", 0.015)
            self.air_mixing_rate_m3_s[idx_a, idx_b] += mixing_rate
            self.air_mixing_rate_m3_s[idx_b, idx_a] += mixing_rate

    def _apply_external_connection(self, connection, standards, building_id):
        idx_a = self.room_indices[f"{building_id}:{connection['from']}"]
        target = connection["to"]
        thermal_code = standards[connection["thermal_code"]]

        windows_to_solve = []
        windows_area_sum_m2 = 0

        for window in connection.get("windows", []):
            window_standard = standards[window["thermal_code"]]
            windows_area_sum_m2 += window["area"]
            windows_to_solve.append(
                {"area_m2": window["area"], "shgc": window_standard["shgc"]}
            )

        if target != "ground":
            ach = standards["ach_wind_coef"]
            vol_m3 = self.volumes_m3[idx_a]

            infiltration_m3_s = (vol_m3 * ach) / 3600
            self.infiltration_rate_m3_s[idx_a] += infiltration_m3_s

            self.external_connections.append(
                {
                    "room_idx": idx_a,
                    "azimuth": connection["azimuth"],
                    "tilt": connection["tilt"],
                    "area_gross_m2": connection["area"],
                    "windows": windows_to_solve,
                    "volume_m3": vol_m3,
                    "ach_wind_coef": ach,
                    "u_value_w_m2k": thermal_code["u_value"],
                    "absorptance": thermal_code["absorptance"],
                }
            )

        wall_net_area_m2 = connection["area"] - windows_area_sum_m2

        ua_wall_w_k = wall_net_area_m2 * thermal_code["u_value"]
        ua_windows_w_k = sum(
            win["area"] * standards[win["thermal_code"]]["u_value"]
            for win in connection.get("windows", [])
        )
        ua_total_w_k = ua_wall_w_k + ua_windows_w_k

        if target == "ground":
            self.ext_ground_conductance_w_k[idx_a] += ua_total_w_k
        else:
            self.ext_air_conductance_w_k[idx_a] += ua_total_w_k

        self.heat_capacity_j_k[idx_a] += (
            wall_net_area_m2 * thermal_code["heat_capacity_per_m2"]
        )
