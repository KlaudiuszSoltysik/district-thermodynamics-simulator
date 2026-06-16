import numpy as np
import pandas as pd
from scipy.optimize import linprog

class BessMPC:
    def __init__(self, bess, energy_service, weather_service, pv_farm, heat_pump):
        self.bess = bess
        self.energy_service = energy_service
        self.weather_service = weather_service
        self.pv_farm = pv_farm
        self.heat_pump = heat_pump

    def optimize(self, current_time, dt_seconds, baseline_load_kw, pv_forecast_kw):
        steps = len(baseline_load_kw)
        dt_hours = dt_seconds / 3600.0

        # 1. Pobieranie cen z wyprzedzeniem
        price_forecast = np.zeros(steps)
        sim_time = current_time
        for i in range(steps):
            weather = self.weather_service.get_weather(sim_time)
            costs = self.energy_service.get_effective_costs(
                sim_time, self.pv_farm, self.heat_pump, weather
            )
            price_forecast[i] = costs.electricity_price_eur_per_mwh
            sim_time += pd.Timedelta(seconds=dt_seconds)

        # 2. Budowa Macierzy dla Scipy Linprog (HiGHS solver)
        # Zmienne układamy w długi wektor: 
        # [P_charge_0 ... P_charge_N, P_discharge_0 ... P_discharge_N, E_soc_0 ... E_soc_N]
        # Razem: 3 * steps zmiennych
        
        c = np.zeros(3 * steps)
        bounds = []
        DEGRADATION_PENALTY = 0.01

        # Wypełnianie funkcji celu (koszty) i limitów (min/max mocy)
        for i in range(steps):
            # Koszt ładowania: kupujemy prąd
            c[i] = (price_forecast[i] * dt_hours) + DEGRADATION_PENALTY
            bounds.append((0, self.bess.max_power_kw))
            
            # Koszt rozładowania: sprzedajemy prąd (ujemny koszt = zysk)
            c[steps + i] = (-price_forecast[i] * dt_hours) + DEGRADATION_PENALTY
            bounds.append((0, self.bess.max_power_kw))
            
            # Stan baterii nie generuje bezpośrednio kosztów na giełdzie
            c[2 * steps + i] = 0.0
            bounds.append((self.bess.min_soc * self.bess.capacity_kwh, 
                           self.bess.max_soc * self.bess.capacity_kwh))

        # Macierz równań fizyki baterii (A_eq * x = b_eq)
        # Bilans: E[i] - E[i-1] - (Charge * sprawność) + (Discharge / sprawność) = 0
        A_eq = np.zeros((steps, 3 * steps))
        b_eq = np.zeros(steps)
        eff = self.bess.efficiency
        
        for i in range(steps):
            A_eq[i, i] = -dt_hours * eff                 # Współczynnik dla P_charge
            A_eq[i, steps + i] = dt_hours / eff          # Współczynnik dla P_discharge
            A_eq[i, 2 * steps + i] = 1.0                 # Współczynnik dla E_soc[i]
            
            if i == 0:
                # Dla pierwszej godziny używamy aktualnego stanu baterii
                b_eq[i] = self.bess.current_soc * self.bess.capacity_kwh
            else:
                A_eq[i, 2 * steps + i - 1] = -1.0        # Odejmujemy poprzedni stan E_soc[i-1]
                b_eq[i] = 0.0

        # Nierówność wymuszająca zrzut prądu pod koniec horyzontu (Terminal Constraint)
        A_ub = np.zeros((1, 3 * steps))
        b_ub = np.zeros(1)
        
        A_ub[0, 3 * steps - 1] = 1.0 # Ostatni element E_soc
        b_ub[0] = (self.bess.min_soc + 0.05) * self.bess.capacity_kwh

        # 3. Rozwiązanie (Używamy super-szybkiego silnika 'highs')
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs'
        )

        optimal_plan_kw = np.zeros(steps)
        internal_prices = np.copy(price_forecast)

        if res.success:
            # Wyciągamy wektory ładowania i rozładowania, łącząc je w jeden plan mocy
            p_charge_opt = res.x[0:steps]
            p_discharge_opt = res.x[steps:2*steps]
            optimal_plan_kw = p_charge_opt - p_discharge_opt
            
            # Wystawienie darmowych cen dla HVAC, gdy bateria oddaje prąd
            for i in range(steps):
                if optimal_plan_kw[i] < -0.1:
                    internal_prices[i] = 0.0
        else:
            print("Linprog failed to converge!")

        return optimal_plan_kw, internal_prices