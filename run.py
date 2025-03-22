import os
import numpy as np
import torch
from tqdm import tqdm  # Pasek postępu
from env import TradingEnv
from agent import DQNAgent
from utils import load_and_process_data, plot_results

# 📊 Załaduj dane rynkowe
prices, log_returns = load_and_process_data('data/etf_prices.csv')

# 🏦 Inicjalizacja środowiska
env = TradingEnv(prices, log_returns)
state_size = env._get_state().shape[0]
action_size = env.n_assets  

# 🔄 Ścieżki do modelu i epsilon
MODEL_PATH = "dqn_trading_model.pth"
EPSILON_PATH = "epsilon.txt"

# 📌 Tworzymy agenta
agent = DQNAgent(state_size, action_size)

# 🔄 **Wczytujemy istniejący model**
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    print("✅ Model wczytany!")
else:
    print("⚠️ Brak zapisanego modelu. Trening od zera.")

# 🔄 **Wczytujemy epsilon, jeśli istnieje**
if os.path.exists(EPSILON_PATH):
    with open(EPSILON_PATH, "r") as f:
        agent.epsilon = float(f.read().strip())
    print(f"🔄 Kontynuujemy trening z epsilon = {agent.epsilon:.4f}")
else:
    agent.epsilon = 1.0  # Pierwszy trening

# 📌 Parametry treningowe
EPISODES = 2000
EPISODE_LENGTH = 252

# 📊 Zapis wyników
all_rewards = []
all_portfolio_values = []
all_epsilons = []

# 🚀 Trening
progress_bar = tqdm(range(EPISODES), desc="📈 Trening modelu", unit="epizod")

for episode in progress_bar:
    state = env.reset(episode_length=EPISODE_LENGTH)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay()  # Trening modelu

    # 📊 Zapis wyników epizodu
    all_rewards.append(total_reward)
    all_portfolio_values.append(env._get_portfolio_value())
    all_epsilons.append(agent.epsilon)

    # 🏆 Co 10 epizodów aktualizujemy target network
    if episode % 10 == 0:
        agent.update_target_network()

    # 🔄 Aktualizacja paska postępu
    progress_bar.set_postfix({
        "🎯 Nagroda": f"{total_reward:.2f}",
        "💰 Portfel": f"{all_portfolio_values[-1]:.2f}",
        "🧠 Epsilon": f"{agent.epsilon:.4f}"
    })

# 🏆 Po treningu zapisujemy model i epsilon
agent.save_model(MODEL_PATH)

with open(EPSILON_PATH, "w") as f:
    f.write(str(agent.epsilon))

print(f"✅ Model i epsilon zapisane!")

# 📊 Wizualizacja wyników
plot_results(all_rewards, all_portfolio_values, all_epsilons, None)
