import numpy as np

from shared.MongoDbController import MongoDbController


class MeteringService:
    def __init__(self, A, num_nodes, district_id_dict):
        self.A = A
        self.num_nodes = num_nodes
        self.district_id_dict = district_id_dict

        self.mongodb = MongoDbController()

        self.is_enabled_mask = None

        self.energy_costs = {}

        self.total_elec_import = 0.0
        self.total_elec_export = 0.0
        self.total_pv_yield = 0.0
        self.total_gas_import = 0.0

        self.admin_elec_cost = 0.0
        self.admin_gas_cost = 0.0

        self.admin_elec_revenue = 0.0
        self.total_tenant_revenue = 0.0

        self.room_heat_delivered = {
            self.district_id_dict[i]: 0.0 for i in range(self.num_nodes)
        }
        self.room_cool_delivered = {
            self.district_id_dict[i]: 0.0 for i in range(self.num_nodes)
        }
        self.room_vent_volume = {
            self.district_id_dict[i]: 0.0 for i in range(self.num_nodes)
        }
        self.room_energy_usage = {
            self.district_id_dict[i]: 0.0 for i in range(self.num_nodes)
        }

        self.room_billing_cost = {
            self.district_id_dict[i]: 0.0 for i in range(self.num_nodes)
        }

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

    def update_meters(self, current_time, dt, energy_costs, q_hvac, v_hvac):
        self.energy_costs = energy_costs
        hours = dt / 3600.0

        current_h = current_time.hour
        current_mask = self.is_enabled_mask[:, current_h]

        q_heating = np.maximum(0, q_hvac)
        q_cooling = np.maximum(0, -q_hvac)

        heat_elec = np.sum(q_heating) / energy_costs["cop_heating"] / 1000.0
        cool_elec = np.sum(q_cooling) / energy_costs["cop_cooling"] / 1000.0

        # TODO: maybe change that later
        vent_elec = np.sum(v_hvac)

        base_power_kw = self.A * 0.002
        active_power_kw = self.A * 0.008 * current_mask
        plugs_power_kw = base_power_kw + active_power_kw

        total_plugs_elec = np.sum(plugs_power_kw)

        total_elec = heat_elec + cool_elec + vent_elec + total_plugs_elec

        grid_buy = np.maximum(0, total_elec - energy_costs["pv_farm_yield"])
        grid_sell = np.maximum(0, energy_costs["pv_farm_yield"] - total_elec)

        gas_buy = 0.0

        self.total_pv_yield += energy_costs["pv_farm_yield"] * hours
        self.total_elec_import += grid_buy * hours
        self.total_elec_export += grid_sell * hours

        self.admin_elec_cost += (grid_buy * hours) * (
            energy_costs["electricity_price"] / 1000.0
        )

        self.total_gas_import += gas_buy * hours
        self.admin_gas_cost += (gas_buy * hours) * (energy_costs["gas_price"] / 1000.0)

        tenant_tariff = 0.35
        export_tariff = energy_costs["electricity_price"]

        self.admin_elec_revenue += (grid_sell * hours) * (export_tariff / 1000.0)

        for i in range(self.num_nodes):
            room_id = self.district_id_dict[i]
            power = q_hvac[i]
            vent = v_hvac[i]

            room_plugs_kw = plugs_power_kw[i]
            room_elec_kw = 0.0

            if power > 0:
                self.room_heat_delivered[room_id] += (power / 1000.0) * hours
                room_elec_kw += (power / energy_costs["cop_heating"]) / 1000.0
            elif power < 0:
                self.room_cool_delivered[room_id] += (abs(power) / 1000.0) * hours
                room_elec_kw += (abs(power) / energy_costs["cop_cooling"]) / 1000.0

            self.room_vent_volume[room_id] += vent * hours
            room_elec_kw += vent

            self.room_energy_usage[room_id] += room_plugs_kw * hours
            room_elec_kw += room_plugs_kw

            step_room_energy_kwh = room_elec_kw * hours
            step_room_cost = step_room_energy_kwh * tenant_tariff

            self.room_billing_cost[room_id] += step_room_cost
            self.total_tenant_revenue += step_room_cost

    def get_meter_readings(self):
        cost_margin = (self.admin_elec_revenue + self.total_tenant_revenue) - (
            self.admin_elec_cost + self.admin_gas_cost
        )

        return {
            "admin_meters": {
                "electricity_import": round(self.total_elec_import, 3),
                "electricity_export": round(self.total_elec_export, 3),
                "pv_farm_yield": round(self.total_pv_yield, 3),
                "gas_import": round(self.total_gas_import, 3),
                "electricity_cost": round(self.admin_elec_cost, 2),
                "gas_cost": round(self.admin_gas_cost, 2),
                "admin_electricity_revenue": round(self.admin_elec_revenue, 2),
                "tenant_billing_revenue": round(self.total_tenant_revenue, 2),
                "cost_margin": round(cost_margin, 2),
                "electricity_price": round(
                    self.energy_costs.get("electricity_price", 0.0), 2
                ),
                "gas_price": round(self.energy_costs.get("gas_price", 0.0), 2),
            },
            "tenant_meters": {
                "heating": {
                    k: round(v, 3) for k, v in self.room_heat_delivered.items()
                },
                "cooling": {
                    k: round(v, 3) for k, v in self.room_cool_delivered.items()
                },
                "ventilation": {
                    k: round(v, 2) for k, v in self.room_vent_volume.items()
                },
                "energy_usage": {
                    k: round(v, 2) for k, v in self.room_energy_usage.items()
                },
                "billing_cost": {
                    k: round(v, 2) for k, v in self.room_billing_cost.items()
                },
            },
        }
