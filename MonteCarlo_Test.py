import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from env import TradingEnv
from agent import DQNAgent
from utils import load_and_process_data

# 📊 Załaduj rzeczywiste ceny i logarytmiczne zwroty
prices, log_returns = load_and_process_data('data/etf_prices_test_downward.csv')

# 🏦 Inicjalizacja środowiska testowego
env = TradingEnv(prices, log_returns)
state_size = env._get_state().shape[0]
action_size = env.n_assets  

# 🔄 Wczytaj wytrenowany model
MODEL_PATH = "dqn_trading_model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Brak modelu: {MODEL_PATH}. Najpierw wytrenuj model!")

agent = DQNAgent(state_size, action_size)
agent.load_model(MODEL_PATH)

# 🚀 Wyłączamy eksplorację (test = tylko strategia modelu)
agent.epsilon = 0.0  

# 🔄 Parametry testu Monte Carlo
TEST_DAYS = 252 # 🏆 Liczba dni testowych
N_SIMULATIONS = 100  # 🏆 Liczba losowych testów

# 📊 Inicjalizacja wyników Monte Carlo
best_value = float('-inf')
worst_value = float('inf')
best_trades = None
worst_trades = None
final_portfolio_values = []
total_rewards = []

# 🔄 Symulacje Monte Carlo
for sim in range(N_SIMULATIONS):
    START_INDEX = np.random.randint(0, len(prices) - TEST_DAYS)

    # 🔄 Reset środowiska przed każdą symulacją
    env.start_step = START_INDEX
    env.current_step = START_INDEX
    env.episode_length = TEST_DAYS
    state = env.reset(episode_length=TEST_DAYS)
    
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)  
        next_state, reward, done = env.step(action)

        state = next_state
        total_reward += reward

    final_value = env._get_portfolio_value()
    final_portfolio_values.append(final_value)
    total_rewards.append(total_reward)

    # 🔝 Sprawdzenie najlepszej symulacji
    if final_value > best_value:
        best_value = final_value
        best_trades = env.trades.copy()  # ✅ Zapisujemy kopię listy transakcji

    # 🔻 Sprawdzenie najgorszej symulacji
    if final_value < worst_value:
        worst_value = final_value
        worst_trades = env.trades.copy()  # ✅ Zapisujemy kopię listy transakcji

    print(f"🔄 Symulacja {sim+1}/{N_SIMULATIONS} | 🎯 Nagroda: {total_reward:.2f} | 💰 Portfel: {final_value:.2f}")

# 📊 Podsumowanie Monte Carlo
mean_value = np.mean(final_portfolio_values)
std_value = np.std(final_portfolio_values)
mean_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)

print("\n📊 Wyniki Monte Carlo:")
print(f"💰 Średnia końcowa wartość portfela: ${mean_value:.2f} ± {std_value:.2f}")
print(f"🎯 Średnia nagroda: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"🏆 Najlepsza symulacja: {best_value:.2f} | 📉 Najgorsza symulacja: {worst_value:.2f}")

# 🏆 Zapisujemy tylko najlepszą i najgorszą symulację
def save_trades(trades, filename):
    if trades:
        df = pd.DataFrame(trades, columns=["Date", "ETF_ID", "Action", "Price", "Shares", "Cash Before", "Cash After", "Portfolio Value", "Action Value", "Currently Held"])
        df = df.drop_duplicates()  # ✅ Usuwamy duplikaty przed zapisem
        df.to_csv(filename, index=False)
        print(f"✅ Transakcje zapisane: {filename}")

save_trades(best_trades, "./monte_carlo/MonteCarlo_Best.csv")
save_trades(worst_trades, "./monte_carlo/MonteCarlo_Worst.csv")

# 📈 Wizualizacja wyników Monte Carlo
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 📉 Histogram końcowych wartości portfela
axes[0].hist(final_portfolio_values, bins=20, color="blue", alpha=0.7, edgecolor="black")
axes[0].set_title("Histogram końcowej wartości portfela")
axes[0].set_xlabel("Wartość portfela ($)")
axes[0].set_ylabel("Liczba symulacji")

# 📊 Histogram nagród
axes[1].hist(total_rewards, bins=20, color="green", alpha=0.7, edgecolor="black")
axes[1].set_title("Histogram sumarycznej nagrody")
axes[1].set_xlabel("Nagroda")
axes[1].set_ylabel("Liczba symulacji")

plt.tight_layout()
plt.show()
