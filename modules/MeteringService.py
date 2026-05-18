import numpy as np


class MeteringService:
    TENANT_TARIF_EUR_PER_KWH = 0.35

    def __init__(self, num_nodes, district_id_dict):
        self.num_nodes = num_nodes
        self.district_id_dict = district_id_dict

        self.energy_costs = {}

        self.total_elec_import_kwh = 0
        self.total_elec_export_kwh = 0
        self.total_pv_yield_kwh = 0
        self.total_pv_self_consumed_kwh = 0
        self.total_gas_import_kwh = 0

        self.admin_elec_cost = 0
        self.admin_gas_cost = 0
        self.admin_elec_revenue = 0
        self.total_tenant_revenue = 0

        self.room_heat_delivered_kwh = {
            self.district_id_dict[i]: 0 for i in range(self.num_nodes)
        }
        self.room_cool_delivered_kwh = {
            self.district_id_dict[i]: 0 for i in range(self.num_nodes)
        }
        self.room_vent_volume_m3 = {
            self.district_id_dict[i]: 0 for i in range(self.num_nodes)
        }
        self.room_billing_cost = {
            self.district_id_dict[i]: 0 for i in range(self.num_nodes)
        }

    def update_meters(self, dt_seconds, energy_costs, q_hvac_w, v_vent_m3_s):
        self.energy_costs = energy_costs
        hours = dt_seconds / 3600

        q_heating_w = np.maximum(0, q_hvac_w)
        q_cooling_w = np.maximum(0, -q_hvac_w)

        heat_elec_kw = np.sum(q_heating_w) / energy_costs.cop_heating / 1000
        cool_elec_kw = np.sum(q_cooling_w) / energy_costs.cop_cooling / 1000

        # TODO: change that
        vent_elec_kw = np.sum(v_vent_m3_s)

        total_elec_demand_kw = heat_elec_kw + cool_elec_kw + vent_elec_kw
        pv_yield_kw = energy_costs.pv_yield_kw

        grid_buy_kw = np.maximum(0, total_elec_demand_kw - pv_yield_kw)
        grid_sell_kw = np.maximum(0, pv_yield_kw - total_elec_demand_kw)
        pv_self_consumed_kw = min(total_elec_demand_kw, pv_yield_kw)

        gas_buy_kw = 0

        self.total_pv_yield_kwh += pv_yield_kw * hours
        self.total_elec_import_kwh += grid_buy_kw * hours
        self.total_elec_export_kwh += grid_sell_kw * hours
        self.total_pv_self_consumed_kwh += pv_self_consumed_kw * hours
        self.total_gas_import_kwh += gas_buy_kw * hours

        self.admin_elec_cost += (grid_buy_kw * hours) * (
            energy_costs.electricity_price_eur_per_mwh / 1000
        )
        self.admin_gas_cost += (gas_buy_kw * hours) * (
            energy_costs.gas_price_eur_per_mwh / 1000
        )

        export_tariff = energy_costs.electricity_price_eur_per_mwh

        self.admin_elec_revenue += (grid_sell_kw * hours) * (export_tariff / 1000)

        for i in range(self.num_nodes):
            room_id = self.district_id_dict[i]
            power_w = q_hvac_w[i]
            vent_m3_s = v_vent_m3_s[i]

            room_elec_kw = 0

            if power_w > 0:
                self.room_heat_delivered_kwh[room_id] += (power_w / 1000) * hours
                room_elec_kw += (power_w / energy_costs.cop_heating) / 1000
            elif power_w < 0:
                self.room_cool_delivered_kwh[room_id] += (abs(power_w) / 1000) * hours
                room_elec_kw += (abs(power_w) / energy_costs.cop_cooling) / 1000

            self.room_vent_volume_m3[room_id] += vent_m3_s * 3600 * hours
            room_elec_kw += vent_m3_s

            step_room_energy_kwh = room_elec_kw * hours
            step_room_cost = step_room_energy_kwh * self.TENANT_TARIF_EUR_PER_KWH

            self.room_billing_cost[room_id] += step_room_cost
            self.total_tenant_revenue += step_room_cost

    def get_meter_readings(self):
        cost_margin = (self.admin_elec_revenue + self.total_tenant_revenue) - (
            self.admin_elec_cost + self.admin_gas_cost
        )

        self_consumption_rate = (
            (self.total_pv_self_consumed_kwh / self.total_pv_yield_kwh * 100)
            if self.total_pv_yield_kwh > 0
            else 0
        )
        total_building_usage_kwh = (
            self.total_elec_import_kwh + self.total_pv_self_consumed_kwh
        )
        self_sufficiency_rate = (
            (self.total_pv_self_consumed_kwh / total_building_usage_kwh * 100)
            if total_building_usage_kwh > 0
            else 0
        )

        return {
            "admin_meters": {
                "electricity_import_kwh": self.total_elec_import_kwh,
                "electricity_export_kwh": self.total_elec_export_kwh,
                "pv_self_consumed_kwh": self.total_pv_self_consumed_kwh,
                "pv_self_consumption_rate_pct": self_consumption_rate,
                "pv_self_sufficiency_rate_pct": self_sufficiency_rate,
                "gas_import_kwh": self.total_gas_import_kwh,
                "electricity_cost": self.admin_elec_cost,
                "gas_cost": self.admin_gas_cost,
                "admin_electricity_revenue": self.admin_elec_revenue,
                "tenant_billing_revenue": self.total_tenant_revenue,
                "cost_margin": cost_margin,
            },
            "tenant_meters": {
                "heating_kwh": {k: v for k, v in self.room_heat_delivered_kwh.items()},
                "cooling_kwh": {k: v for k, v in self.room_cool_delivered_kwh.items()},
                "ventilation_m3": {k: v for k, v in self.room_vent_volume_m3.items()},
                "billing_cost": {k: v for k, v in self.room_billing_cost.items()},
            },
        }
