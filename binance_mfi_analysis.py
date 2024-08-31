import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from binance.client import Client
from scipy.signal import find_peaks
from functions import load_config, setup_logging, get_1m_candles, calculate_mfi, find_extrema, real_time_extrema, plot_asset, get_candles, MFI_TIMEINTERVAL

def calculate_price_change(candles, minima, maxima):
    changes = []
    for i in range(len(minima)):
        if i < len(maxima):
            min_price = float(candles[minima[i]][4])
            max_price = float(candles[maxima[i]][4])
            changes.append((max_price - min_price) / min_price * 100)
    return changes

def analyze_pair(symbol):
    candles = get_candles(symbol, "1m", "1440")
    if len(candles) == 0:
        return None
    
    mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)
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

def main():
    client = Client()
    # Fetch exchange information
    exchange_info = client.get_exchange_info()

    # Filter symbols that end with 'USDT' and are available for spot trading
    symbols = [
        symbol['symbol'] 
        for symbol in exchange_info['symbols'] 
        if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']
    ]
    
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
    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    df.to_csv(f"out/{filename_suffix}_crypto_mfi_analysis.csv", index=False)
    df.to_excel(f"out/{filename_suffix}_crypto_mfi_analysis.xlsx", index=False)

    # Select top 10 assets based on highest real_time_sum_change
    top_assets = df.nlargest(10, 'real_time_sum_change')
    results_top = [res for res in results if res["symbol"] in list(top_assets["symbol"])]

    # Plotting each of the top 10 assets
    for asset in results_top:
        plot_asset(asset, "_analysis")

if __name__ == "__main__":
    main()
