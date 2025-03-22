import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from env import TradingEnv
from agent import DDPGAgent
from utils import load_and_process_data, plot_results

# 🔧 Hiperparametry
EPISODES = 1000
EPISODE_LENGTH = 252
save_every = 200
early_stopping_patience = 500
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 📊 Dane
prices, log_returns = load_and_process_data('data/etf_prices.csv')

# 🔁 Środowisko
env = TradingEnv(prices, log_returns)
state_size = env._get_state().shape[0]
action_size = env.n_assets

# 🧠 Agent
agent = DDPGAgent(state_size, action_size)

# 🔁 Wczytaj checkpoint jeśli istnieje
ACTOR_PATH = os.path.join(results_dir, "ddpg_actor_ep1000.pth")
CRITIC_PATH = os.path.join(results_dir, "ddpg_critic_ep1000.pth")
if os.path.exists(ACTOR_PATH) and os.path.exists(CRITIC_PATH):
    agent.load_model(ACTOR_PATH, CRITIC_PATH)
    print("✅ Modele wczytane – kontynuujemy trening!")

# 📊 Śledzenie wyników
all_rewards = []
all_portfolio_values = []
best_value = -np.inf
no_improvement = 0

# 📝 CSV log
log_path = os.path.join(results_dir, "ddpg_train_log.csv")
log_df = pd.DataFrame(columns=["episode", "reward", "portfolio_value"])
log_df.to_csv(log_path, index=False)

# 🚀 Trening
progress_bar = tqdm(range(EPISODES), desc="Trening DDPG", unit="epizod")

for episode in progress_bar:
    state = env.reset(episode_length=EPISODE_LENGTH)
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state, add_noise=True)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

    final_value = env._get_portfolio_value()
    all_rewards.append(total_reward)
    all_portfolio_values.append(final_value)

    # 📈 CSV log
    pd.DataFrame([[episode, total_reward, final_value]],
                 columns=["episode", "reward", "portfolio_value"]).to_csv(
        log_path, mode="a", header=False, index=False
    )

    # 🛑 Early stopping
    if final_value > best_value:
        best_value = final_value
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement >= early_stopping_patience:
        print(f"⏹️ EARLY STOPPING – brak poprawy przez {early_stopping_patience} epizodów.")
        break

    # 💾 Checkpoint
    if (episode + 1) % save_every == 0:
        actor_ckpt = os.path.join(results_dir, f"ddpg_actor_ep{episode+1}.pth")
        critic_ckpt = os.path.join(results_dir, f"ddpg_critic_ep{episode+1}.pth")
        agent.save_model(actor_ckpt, critic_ckpt)

    # 🔁 Pasek postępu
    progress_bar.set_postfix({
        "🎯 Nagroda": f"{total_reward:.2f}",
        "💰 Portfel": f"{final_value:.2f}",
        "🛑 Brak poprawy": f"{no_improvement}"
    })

# 🔚 Zapis ostatecznych wag
agent.save_model(ACTOR_PATH, CRITIC_PATH)
print("✅ Trening zakończony, model zapisany!")

# 📈 Wykres końcowy
plot_results(all_rewards, all_portfolio_values, epsilons=None, actions=None)
