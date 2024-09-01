import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from binance.client import Client
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import yaml

MFI_THRESHOLD_LOW = 20
MFI_THRESHOLD_HIGH = 80
MFI_STEP_THRESHOLD = 2
MFI_TIMEINTERVAL = 14
MFI_TRADING_TIMEOUT_H = 12

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(file_suffix=""):
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{file_suffix}{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

def get_candles(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    return response.json()

def get_1m_candles(symbol):
    get_candles(symbol, "1m", "1440")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1440" # 24h
    response = requests.get(url)
    return response.json()

def calculate_mfi(candles, timeperiod=14):
    high = np.array([float(c[2]) for c in candles])
    low = np.array([float(c[3]) for c in candles])
    close = np.array([float(c[4]) for c in candles])
    volume = np.array([float(c[5]) for c in candles])
    return ta.MFI(high, low, close, volume, timeperiod=timeperiod)

def find_extrema(data):
    peaks, _ = find_peaks(data, distance=30)
    troughs, _ = find_peaks(-data, distance=30)
    return troughs, peaks

def real_time_extrema(mfi):
    buy_signals = []
    sell_signals = []

    last_local_minima = 100
    candles_above_threshold = 0

    bought = False
    
    for i in range(1, len(mfi)):
        mfi_i = mfi[i]
        
        # minima
        if mfi_i < MFI_THRESHOLD_LOW and mfi_i < last_local_minima:
            last_local_minima = mfi_i

        diff_to_minima = mfi_i - last_local_minima 
        if mfi_i < MFI_THRESHOLD_LOW and mfi_i > last_local_minima and diff_to_minima > MFI_STEP_THRESHOLD and not bought:
            last_local_minima = 100
            bought = True
            buy_signals.append(i) # buy signal
        
        if not bought:
            continue

        if mfi_i > MFI_THRESHOLD_HIGH:
            candles_above_threshold += 1
        else:
            candles_above_threshold = 0

        # maxima
        if candles_above_threshold > 1:
            # sell as soon as MFI stays above threshold for 2 candles
            candles_above_threshold = 0
            bought = False
            sell_signals.append(i) # sell signal

    return buy_signals, sell_signals


def plot_asset(asset_data, plot_suffix=""):
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the candlestick chart on the first subplot
    mpf.plot(df, type='line', ax=ax1, volume=False, show_nontrading=True)

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
    plt.savefig(f'out/{symbol}_chart{plot_suffix}.png')
    # plt.show()
    plt.close()
