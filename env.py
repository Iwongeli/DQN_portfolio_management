import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self, prices, log_returns, initial_balance=100000, transaction_cost=0.01, lookback=252, eta=1/252):
        """
        Środowisko handlu ETF-ami oparte na Reinforcement Learning.
        
        :param prices: rzeczywiste ceny akcji (DataFrame)
        :param log_returns: zwroty logarytmiczne (DataFrame)
        """
        self.prices = prices  # 📊 Ceny do transakcji
        self.log_returns = log_returns  # 🔥 Zwroty logarytmiczne do przestrzeni stanów
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.n_assets = prices.shape[1]  # Liczba aktywów
        self.lookback = lookback  # Długość historii dla agenta
        self.eta = eta  # Współczynnik wygładzania DSR
        
        # 🔄 DSR - Inicjalizacja A_t i B_t
        self.A_t = 0  # Eksponencjalnie wygładzona średnia zwrotów
        self.B_t = 0  # Eksponencjalnie wygładzona wariancja zwrotów
        
        print(f"🔍 LICZBA ETF-ÓW: {self.n_assets}")  
        self.reset()

    def step(self, action):
        previous_value = self._get_portfolio_value()
        prices = self.prices.iloc[self.current_step, :self.n_assets].values
        date = self.prices.index[self.current_step]

        for i in range(self.n_assets):
            if prices[i] == 0 or np.isnan(prices[i]):
                continue

            # ✅ **Nowa logika: Konwersja wartości akcji na rzeczywiste jednostki kupna/sprzedaży**
            action_scaled = action[i] * self.balance / prices[i]
            action_scaled = int(round(action_scaled, 0))  # 🔄 Zaokrąglenie do pełnych jednostek akcji

            cash_before = self.balance  

            if action[i] > 0.1:  # 🟢 **KUPNO**
                max_shares = self.balance / (prices[i] * (1 + self.transaction_cost))
                num_shares = int(min(action_scaled, max_shares))  # 🏆 Całkowita liczba akcji
                cost = (num_shares * prices[i] * (1 + self.transaction_cost)).round(3)

                if cost <= self.balance:
                    self.portfolio[i] += num_shares
                    self.balance = round(self.balance - cost, 3)
                    action_type = "BUY"
                else:
                    action_type = "BUY_FAILED"
                    num_shares = 0  

            elif action[i] < -0.1:  # 🔴 **SPRZEDAŻ**
                action_scaled  = -action[i] * self.portfolio[i]  # odniesienie do posiadanych jednostek
                num_shares = int(min(action_scaled, self.portfolio[i]))  # 🏆 Całkowita liczba akcji

                if num_shares > 0:
                    self.balance = round(self.balance + (num_shares * prices[i] * (1 - self.transaction_cost)), 3) 
                    self.portfolio[i] -= num_shares
                    action_type = "SELL"
                else:
                    action_type = "HOLD"
                    num_shares = 0  

            else:  # ⚪ **HOLD**
                action_type = "HOLD"
                num_shares = 0  

            cash_after = self.balance  
            portfolio_value = self._get_portfolio_value()  # 📊 Wartość portfela po transakcji

            # 📝 **Zapisujemy transakcję**
            self.trades.append((date, i, action_type, prices[i], num_shares, cash_before, cash_after, portfolio_value, action[i], self.portfolio[i]))

        self.current_step += 1
        done = self.current_step >= self.start_step + self.episode_length  
        new_value = self._get_portfolio_value()

        # 🔥 **Obliczamy zwrot logarytmiczny portfela**
        R_t = np.log(new_value / previous_value) if previous_value > 0 else 0

        # 🔄 **Aktualizacja A_t i B_t**
        self.A_t = self.A_t + self.eta * (R_t - self.A_t)
        self.B_t = self.B_t + self.eta * (R_t**2 - self.B_t)

        # 🔢 **Obliczamy funkcję nagrody - DSR**
        numerator = (self.B_t - self.eta * R_t**2) * (R_t - self.A_t) - 0.5 * self.A_t * (R_t**2 - self.B_t)
        denominator = (self.B_t - self.A_t**2) ** (3 / 2)
        dsr = numerator / denominator if denominator > 0 else 0  # 📊 Zapobieganie dzieleniu przez zero
 
        alfa = 0.1
        
        # 🎯 Ustawiamy nagrodę równą zwrotowi
        reward = R_t + dsr * alfa


        self.history.append(new_value)

        if self.balance < 0:
            print(f"🚨 {date}: NEGATYWNE saldo gotówki: {self.balance:.2f} | 📈 Wartość portfela: {new_value:.2f}")

        return self._get_state(), reward, done



    def reset(self, episode_length=None):
        self.balance = self.initial_balance
        self.portfolio = np.zeros(self.n_assets)
        self.episode_length = episode_length if episode_length else 100  
        
        max_start = len(self.prices) - self.episode_length  
        self.start_step = np.random.randint(self.lookback, max_start) if max_start > self.lookback else self.lookback
        self.current_step = self.start_step

        self.history = []
        self.trades = []

        # 🏆 Resetujemy DSR
        self.A_t = 0
        self.B_t = 0

        return self._get_state()


    def _get_portfolio_value(self):
        """Oblicza wartość portfela na podstawie rzeczywistych cen."""
        prices = self.prices.iloc[self.current_step, :].values
        prices = np.where(prices < 0.01, 0.01, prices)  
        portfolio_value = np.sum(self.portfolio * prices)
        return round(self.balance + portfolio_value, 2)  

    def _get_state(self):
        """
        Tworzy wektor stanu:
        - Znormalizowane zwroty logarytmiczne z ostatnich `lookback` dni (flatten T x N_assets)
        - Udział każdego aktywa i gotówki jako % wartości portfela
        """
        if self.current_step < self.lookback:
            print(f"⚠️ Uwaga: self.current_step = {self.current_step} jest mniejsze niż lookback = {self.lookback}!")
            padding = np.zeros((self.lookback - self.current_step, self.n_assets))
            past_returns = np.vstack([padding, self.log_returns.iloc[:self.current_step].values])
        else:
            past_returns = self.log_returns.iloc[self.current_step - self.lookback:self.current_step].values

        # 📈 Spłaszczamy (T * N_assets) do 1D wektora
        past_returns_flat = past_returns.flatten()

        # 🔄 Standaryzacja (normalizacja) log-returns
        mean = np.mean(past_returns_flat)
        std = np.std(past_returns_flat)
        past_returns_normalized = (past_returns_flat - mean) / (std + 1e-8)

        # 💰 Obliczenie udziałów procentowych w portfelu
        prices = self.prices.iloc[self.current_step].values
        prices = np.where(prices < 0.01, 0.01, prices)  # Zapobieganie błędom przy cenach bliskich zeru

        asset_values = self.portfolio * prices
        total_portfolio_value = self.balance + np.sum(asset_values)

        asset_weights = asset_values / total_portfolio_value
        cash_weight = self.balance / total_portfolio_value

        portfolio_info = np.hstack([asset_weights, cash_weight])  # N_assets + 1

        # 🔗 Połączenie wszystkiego w wektor stanu
        state = np.hstack([past_returns_normalized, portfolio_info])

        expected_size = (self.lookback * self.n_assets) + (self.n_assets + 1)
        assert state.shape[0] == expected_size, \
            f"BŁĄD: Oczekiwano {expected_size}, otrzymano {state.shape[0]}"

        return state.astype(np.float32)


