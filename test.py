import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from env import TradingEnv
from agent import DDPGAgent
from utils import load_and_process_data, backtest_buy_and_hold, backtest_rebalance_sharpe

# ✅ Wczytaj ścieżkę pliku z terminala
if len(sys.argv) < 2:
    print("❌ Podaj ścieżkę do pliku z danymi jako argument: python test.py data/plik.csv")
    sys.exit(1)

data_path = sys.argv[1]

if not os.path.exists(data_path):
    print(f"❌ Plik nie istnieje: {data_path}")
    sys.exit(1)

# 📂 Załaduj dane z podanej ścieżki
prices, log_returns = load_and_process_data(data_path)

# 🧠 Inicjalizacja środowiska testowego
env = TradingEnv(prices, log_returns)
state_size = env._get_state().shape[0]
action_size = env.n_assets

# 📦 Wczytanie wytrenowanych modeli
ACTOR_PATH = "results/ddpg_actor_ep200.pth"
CRITIC_PATH = "results/ddpg_critic_ep200.pth"
if not os.path.exists(ACTOR_PATH) or not os.path.exists(CRITIC_PATH):
    raise FileNotFoundError("❌ Brak modeli: ddpg_actor.pth lub ddpg_critic.pth")

agent = DDPGAgent(state_size, action_size)
agent.load_model(ACTOR_PATH, CRITIC_PATH)

# 📅 Parametry testu
START_INDEX = 0
TEST_DAYS = min(len(prices), 10000)
WINDOW = 252  # Taki sam jak przy treningu

# 🔄 Reset środowiska do testowania
env.start_step = START_INDEX
env.current_step = START_INDEX
env.episode_length = TEST_DAYS
state = env.reset(episode_length=TEST_DAYS)
done = False
total_reward = 0

# 🚀 Przechodzimy przez dane testowe
while not done:
    if env.current_step >= len(prices) - 1:
        break

    action = agent.act(state, add_noise=False)  # bez szumu!
    next_state, reward, done = env.step(action)

    state = next_state
    total_reward += reward

# ✅ Po zakończeniu testu
final_value = env._get_portfolio_value()
test_trades = env.trades

# 📝 Zapis transakcji do CSV
trades_df = pd.DataFrame(
    test_trades,
    columns=[
        "Date", "ETF_ID", "Action", "Price", "Shares",
        "Cash Before", "Cash After", "Portfolio Value", "Action Value", "Currently Held"
    ]
)
os.makedirs("results", exist_ok=True)
trades_df.to_csv("./results/full_test_trades_ddpg.csv", index=False)

# 📊 Wyniki
print("\n📊 Wyniki testu DDPG na pełnych danych:")
print(f"💰 Końcowa wartość portfela: ${final_value:.2f}")
print(f"🎯 Sumaryczna nagroda: {total_reward:.2f}")
print("✅ Pełna historia transakcji zapisana jako `full_test_trades_ddpg.csv`")

# 📈 Wykres porównawczy
dates = prices.index[env.start_step:env.start_step + len(env.history)]

plt.figure(figsize=(12, 6))
plt.plot(dates, env.history, label="DDPG", color="blue")

# 📉 Benchmarki
buy_hold_curve = backtest_buy_and_hold(prices, window=WINDOW)
rebalance_curve = backtest_rebalance_sharpe(prices, window=WINDOW)

cut_start = WINDOW
plot_dates = prices.index[cut_start:cut_start + len(env.history[cut_start:])]

plt.plot(buy_hold_curve.index[cut_start:], buy_hold_curve.values[cut_start:] * env.initial_balance,
         label="Buy & Hold (Sharpe)", linestyle="--", color="steelblue")
plt.plot(rebalance_curve.index[cut_start:], rebalance_curve.values[cut_start:] * env.initial_balance,
         label="Rebalans co 30 dni (Sharpe)", linestyle="-.", color="orange")
plt.axhline(y=env.initial_balance, linestyle="--", color="gray", label="Kapitał początkowy")

plt.title("Porównanie strategii portfelowych (DDPG vs benchmarki)")
plt.xlabel("Data")
plt.ylabel("Wartość portfela ($)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
