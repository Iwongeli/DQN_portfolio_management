import numpy as np
import pandas as pd
import yfinance as yf
import os

# Folder na dane
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_etf_data_to_csv(etfs, start_date, end_date, file_name):
    """
    Pobiera ceny zamknięcia dla listy ETF-ów w określonym zakresie dat i zapisuje do pliku CSV.
    
    Args:
        etfs (list): Lista symboli ETF.
        start_date (str): Data początkowa w formacie 'YYYY-MM-DD'.
        end_date (str): Data końcowa w formacie 'YYYY-MM-DD'.
        file_name (str): Nazwa pliku CSV do zapisania danych.
    """
    all_data = []
    
    for etf in etfs:
        try:
            print(f"Pobieranie danych dla {etf}...")
            data = yf.download(etf, start=start_date, end=end_date, progress=False)[['Close']]
            
            if data.empty:
                print(f"Brak danych dla {etf}, pomijam...")
                continue
            
            data.rename(columns={'Close': etf}, inplace=True)
            all_data.append(data)
        except Exception as e:
            print(f"Błąd pobierania {etf}: {e}")
    
    if not all_data:
        print("Nie udało się pobrać żadnych danych!")
        return
    
    # Połączenie wszystkich pobranych danych w jeden DataFrame
    df = pd.concat(all_data, axis=1)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df = df.round(3)
    file_path = os.path.join(DATA_DIR, file_name)
    
    df.to_csv(file_path, index=False)
    print(f"Dane zostały zapisane do pliku: {file_path}")

# Lista ETF-ów
etfs = ['SPY', 'AGG', 'DBC', 'GLD']
start_date = '2006-01-01'
end_date = '2022-12-31'
file_name = 'etf_prices.csv'
# file_name = 'etf_prices_test_upward.csv'

# Pobranie danych i zapis do CSV
fetch_etf_data_to_csv(etfs, start_date, end_date, file_name)