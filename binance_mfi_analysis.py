import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from binance.client import Client
from scipy.signal import find_peaks

def get_1m_candles(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1440"
    response = requests.get(url)
    return response.json()

def calculate_mfi(candles):
    high = np.array([float(c[2]) for c in candles])
    low = np.array([float(c[3]) for c in candles])
    close = np.array([float(c[4]) for c in candles])
    volume = np.array([float(c[5]) for c in candles])
    return ta.MFI(high, low, close, volume, timeperiod=14)

def find_extrema(data):
    peaks, _ = find_peaks(data, distance=30)
    troughs, _ = find_peaks(-data, distance=30)
    return troughs, peaks

def real_time_extrema(mfi):
    buy_signals = []
    sell_signals = []

    last_local_minima = 100
    last_local_maxima = 0

    bought = False
    
    for i in range(1, len(mfi)):
        mfi_i = mfi[i]
        
        # minima
        if mfi_i < 30 and mfi_i < last_local_minima:
            last_local_minima = mfi_i

        diff_to_minima = mfi_i - last_local_minima 
        if mfi_i > last_local_minima and diff_to_minima > 5 and not bought:
            last_local_minima = 100
            bought = True
            buy_signals.append(i) # buy signal
        
        if not bought:
            continue

        # maxima
        if mfi_i > 70 and mfi_i > last_local_maxima:
            last_local_maxima = mfi_i

        diff_to_maxima = last_local_maxima - mfi_i
        if mfi_i < last_local_maxima and diff_to_maxima > 5:
            last_local_maxima = 0
            bought = False
            sell_signals.append(i) # sell signal

    return buy_signals, sell_signals

def calculate_price_change(candles, minima, maxima):
    changes = []
    for i in range(len(minima)):
        if i < len(maxima):
            min_price = float(candles[minima[i]][4])
            max_price = float(candles[maxima[i]][4])
            changes.append((max_price - min_price) / min_price * 100)
    return changes

def analyze_pair(symbol):
    candles = get_1m_candles(symbol)
    if len(candles) == 0:
        return None
    
    mfi = calculate_mfi(candles)
    troughs, peaks = find_extrema(mfi)
    buy_signals, sell_signals = real_time_extrema(mfi)
    
    optimal_changes = calculate_price_change(candles, troughs, peaks)
    real_time_changes = calculate_price_change(candles, buy_signals, sell_signals)
    
    return {
        "symbol": symbol,
        "optimal_sum_change": sum(optimal_changes),
        "real_time_sum_change": sum(real_time_changes),
        "optimal_vs_real": (sum(real_time_changes) / sum(optimal_changes)) * 100 if sum(optimal_changes) != 0 else 0,
        "24h_volume": candles[-1][7],
        "market_cap": float(candles[-1][4]) * float(candles[-1][5]),
        "vdelta": (float(candles[-1][4]) - float(candles[0][4])) / float(candles[0][4]) * 100
    }

def main():
    client = Client()
    symbols = [symbol['symbol'] for symbol in client.get_all_tickers() if symbol['symbol'].endswith('USDT')]
    
    results = []
    for symbol in tqdm(symbols[1:10]):
        result = analyze_pair(symbol)
        if result:
            results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv("crypto_mfi_analysis.csv", index=False)
    df.to_excel("crypto_mfi_analysis.xlsx", index=False)

if __name__ == "__main__":
    main()
