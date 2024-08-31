import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from binance.client import Client
from scipy.signal import find_peaks
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def setup_logging():
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

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
