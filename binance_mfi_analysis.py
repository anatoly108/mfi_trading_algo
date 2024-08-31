import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from binance.client import Client
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import mplfinance as mpf

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
        if mfi_i < 30 and mfi_i > last_local_minima and diff_to_minima > 2 and not bought:
            last_local_minima = 100
            bought = True
            buy_signals.append(i) # buy signal
        
        if not bought:
            continue

        # maxima
        if mfi_i > 70 and mfi_i > last_local_maxima:
            last_local_maxima = mfi_i

        diff_to_maxima = last_local_maxima - mfi_i
        if mfi_i > 70 and mfi_i < last_local_maxima and diff_to_maxima > 2:
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
        "vdelta": (float(candles[-1][4]) - float(candles[0][4])) / float(candles[0][4]) * 100,
        "candles": candles,
        "mfi": mfi,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals
    }

def plot_asset(asset_data):
    symbol = asset_data["symbol"]
    candles = asset_data["candles"]
    mfi = asset_data["mfi"]
    buy_signals = asset_data["buy_signals"]
    sell_signals = asset_data["sell_signals"]

    # Convert candles to a DataFrame
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    # Create a figure with two subplots (one for the candlestick chart and one for MFI)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the candlestick chart on the first subplot
    mpf.plot(df, type='candle', ax=ax1, volume=False, show_nontrading=True)

    # Plot MFI on the second subplot
    ax2.plot(df.index, mfi, color='blue', label='MFI')

    # Adding buy/sell signals to both charts
    # On the price chart
    ax1.scatter(df.index[buy_signals], df["close"].iloc[buy_signals], color='green', label='Buy Signal', marker='o')
    ax1.scatter(df.index[sell_signals], df["close"].iloc[sell_signals], color='red', label='Sell Signal', marker='o')

    # On the MFI chart
    ax2.scatter(df.index[buy_signals], mfi[buy_signals], color='green', marker='o')
    ax2.scatter(df.index[sell_signals], mfi[sell_signals], color='red', marker='o')

    # Labels and legends
    ax1.set_title(f'{symbol} Price and Buy/Sell Signals')
    ax2.set_title('Money Flow Index (MFI)')

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'out/{symbol}_chart.png')
    # plt.show()

def main():
    client = Client()
    symbols = [symbol['symbol'] for symbol in client.get_all_tickers() if symbol['symbol'].endswith('USDT')]
    
    results = []
    for symbol in tqdm(symbols):  # Limiting to first 10 for testing
        result = analyze_pair(symbol)
        if result:
            results.append(result)
    
    subset_results = [
    {
        "symbol": res["symbol"],
        "optimal_sum_change": res["optimal_sum_change"],
        "real_time_sum_change": res["real_time_sum_change"],
        "optimal_vs_real": res["optimal_vs_real"],
        "24h_volume": res["24h_volume"],
        "market_cap": res["market_cap"],
        "vdelta": res["vdelta"]
    }
        for res in results
    ]

    # Create a DataFrame from the subset results
    df = pd.DataFrame(subset_results)
    df.to_csv("crypto_mfi_analysis.csv", index=False)
    df.to_excel("crypto_mfi_analysis.xlsx", index=False)

    # Select top 10 assets based on highest real_time_sum_change
    top_assets = df.nlargest(10, 'real_time_sum_change')
    results_top = [res for res in results if res["symbol"] in list(top_assets["symbol"])]

    # Plotting each of the top 10 assets
    for asset in results_top:
        plot_asset(asset)

if __name__ == "__main__":
    main()
