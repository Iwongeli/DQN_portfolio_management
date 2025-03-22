import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_process_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # 📊 Ceny do transakcji
    prices = df.copy()

    # 🔥 Obliczamy logarytmiczne zwroty
    log_returns = np.log(df / df.shift(1)).dropna()

    return prices, log_returns


def moving_average(data, window_size=20):
    """Oblicza średnią kroczącą dla podanych danych."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(rewards, portfolio_values, epsilons, actions):
    """Wizualizacja wyników treningu."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    # Wykres wartości portfela
    axes[0].plot(portfolio_values, label="Wartość portfela", color="blue", alpha=0.7)
    
    # Średnia krocząca
    if len(portfolio_values) >= 20:
        ma_values = moving_average(portfolio_values, 20)
        axes[0].plot(range(19, len(portfolio_values)), ma_values, label="Średnia krocząca (20 epiz.)", color="red")
    
    # Linia na poziomie 10 000 USD
    axes[0].axhline(y=10000, color='gray', linestyle='--', label="Poziom 10 000$")
    
    axes[0].set_title("Wartość portfela w czasie")
    axes[0].set_xlabel("Epizod")
    axes[0].set_ylabel("Wartość ($)")
    axes[0].legend()
    
    # Histogram nagród
    axes[1].hist(rewards, bins=20, color="green", alpha=0.7, edgecolor="black")
    axes[1].set_title("Histogram nagród")
    axes[1].set_xlabel("Nagroda")
    axes[1].set_ylabel("Liczba epizodów")
    
    plt.tight_layout()
    plt.show()


def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


def backtest_buy_and_hold(prices, window=252, transaction_cost=0.01):
    returns = np.log(prices / prices.shift(1)).dropna()
    portfolio_value = 1.0

    lookback = returns.iloc[:window]
    sharpe_scores = lookback.mean() / lookback.std()
    sharpe_scores = sharpe_scores[sharpe_scores > 0]

    weights = sharpe_scores / sharpe_scores.sum()

    selected_prices = prices[weights.index]
    log_returns = np.log(selected_prices / selected_prices.shift(1)).dropna()
    log_returns = log_returns.iloc[window:]

    portfolio_returns = log_returns @ weights
    cumulative_returns = np.exp(np.cumsum(portfolio_returns))

    # ➖ Koszt transakcyjny przy zakupie
    cumulative_returns *= (1 - transaction_cost)

    pre_padding = [portfolio_value] * window
    all_values = pre_padding + cumulative_returns.tolist()
    aligned_index = returns.index[:len(all_values)]

    return pd.Series(all_values, index=aligned_index)


def backtest_rebalance_sharpe(prices, rebalance_period=30, window=252, transaction_cost=0.01):
    returns = np.log(prices / prices.shift(1)).dropna()
    values = []
    portfolio_value = 1.0
    current_weights = np.zeros(prices.shape[1])
    previous_weights = np.zeros(prices.shape[1])

    for i in range(window, len(returns)):
        rebalance = False

        if i == window:
            rebalance = True
        elif (i - window) % rebalance_period == 0:
            rebalance = True

        if rebalance:
            lookback = returns.iloc[i - window:i]
            sharpe_scores = lookback.mean() / lookback.std()
            sharpe_scores = sharpe_scores[sharpe_scores > 0]

            weights = sharpe_scores / sharpe_scores.sum()
            previous_weights = current_weights.copy()
            current_weights = np.zeros(prices.shape[1])
            for etf, w in weights.items():
                current_weights[returns.columns.get_loc(etf)] = w

            # ➖ Koszty transakcyjne – suma zmian alokacji
            turnover = np.sum(np.abs(current_weights - previous_weights))
            portfolio_value *= (1 - transaction_cost * turnover)

        daily_return = returns.iloc[i] @ current_weights
        portfolio_value *= np.exp(daily_return)
        values.append(portfolio_value)

    pre_padding = [np.nan] * window
    all_values = pre_padding + values
    aligned_index = returns.index[:len(all_values)]

    return pd.Series(all_values, index=aligned_index)

