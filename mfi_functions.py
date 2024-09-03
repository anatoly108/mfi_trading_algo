import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta, timezone
from binance.client import Client
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import yaml

MFI_THRESHOLD_LOW = 20
MFI_THRESHOLD_LOW_EXTENDED = MFI_THRESHOLD_LOW + 10

MFI_THRESHOLD_HIGH = 80
MFI_THRESHOLD_DECREASE_PER_CANDLE = 2
MFI_THRESHOLD_HIGH_MIN = 50

MFI_STEP_THRESHOLD = 3
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

def convert_to_unix(date_obj):
    return int(date_obj.timestamp() * 1000)

def get_last_complete_time(interval):
    now = datetime.now(timezone.utc)
    
    if interval.endswith('m'):  # Minutes intervals
        minutes = int(interval[:-1])
        last_complete_time = now - timedelta(minutes=now.minute % minutes, seconds=now.second, microseconds=now.microsecond)
    
    elif interval.endswith('h'):  # Hours intervals
        hours = int(interval[:-1])
        last_complete_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % hours)
    
    elif interval == '1d':  # 1 day interval
        last_complete_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif interval == '3d':  # 3 days interval
        last_complete_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(now.day - 1) % 3)
    
    elif interval == '1w':  # 1 week interval
        days_since_monday = now.weekday()
        last_complete_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
    
    elif interval == '1M':  # 1 month interval
        last_complete_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    else:
        raise ValueError("Unsupported interval")
    
    return last_complete_time

# startTime and endTime are datetime objects, easiest way to specify: datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
def get_candles(symbol, interval, startTime=None, endTime=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval
    }
    if startTime is None and endTime is None:
        now = get_last_complete_time(interval)
        params["startTime"] = convert_to_unix(now - timedelta(hours=24))
        params["endTime"] = convert_to_unix(now)
    else:
        startTimeUnix = convert_to_unix(startTime.replace(tzinfo=timezone.utc))
        endTimeUnix = convert_to_unix(endTime.replace(tzinfo=timezone.utc))
        params["startTime"] = startTimeUnix
        params["endTime"] = endTimeUnix
    # TODO: can get TimeoutError: [Errno 60] Operation timed out
    response = requests.get(url, params=params)
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
        
        if mfi_i > MFI_THRESHOLD_LOW:
            last_local_minima = 100

        # minima
        if mfi_i < MFI_THRESHOLD_LOW and mfi_i < last_local_minima:
            last_local_minima = mfi_i

        diff_to_minima = mfi_i - last_local_minima 
        if mfi_i < (MFI_THRESHOLD_LOW + 10) and mfi_i > last_local_minima and diff_to_minima > MFI_STEP_THRESHOLD and not bought:
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


def execute_trade(symbol, quantity, action, config_path, dry_run):
    if dry_run:
        logging.info(f"Dry run {action}")
        return {'price': None}

    config = load_config(config_path)
    client = Client(config['api_key'], config['api_secret'])
    if action == 'buy':
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        logging.info(f"Market Buy Order: {order}")
    elif action == 'sell':
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        logging.info(f"Market Sell Order: {order}")
    
    client.close_connection()

    final_price = np.mean([float(fill['price']) for fill in order['fills']])

    return {'price': final_price}

def get_new_candles_binance_api(symbol, interval, last_candle_timestamp):
    # get many candles just to be sure we didn't miss any due to some glitch
    return(get_candles(symbol, "1m", datetime.fromtimestamp(last_candle_timestamp/1000) - timedelta(minutes=10), get_last_complete_time()))

def run_mfi_trading_algo(symbol, quantity, config_path, dry_run, 
                         negative_cancel_num=3, get_new_candles_function=get_new_candles_binance_api,
                         candles_start_time=None, candles_end_time=None): 
    start_time = datetime.now()
    start_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    # Load initial candles and MFI
    candles = get_candles(symbol, "1m", candles_start_time, candles_end_time)
    mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)

    last_local_minima = 100
    candles_above_threshold = 0
    bought = False
    really_new_candles = []
    total_profit = 0
    buy_signals = []
    sell_signals = []
    profits = []
    candles_since_buy = 0

    # initialize last_local_minima - for the case when there's a buy opportunity immediately at start
    for i in range(1, len(mfi)):
        mfi_i = mfi[i]
        
        # minima
        if mfi_i > MFI_THRESHOLD_LOW:
            # if last candle is above threshold, last_local_minima is reset
            last_local_minima = 100

        if mfi_i < MFI_THRESHOLD_LOW and mfi_i < last_local_minima:
            # if last candle is below threshold, last_local_minima is set accordingly
            last_local_minima = mfi_i
    
    while True:
        # Recalculate MFI with the new candle(s)
        mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)

        # We have to account for the case when we get >1 new candle
        mfi_new_from = len(mfi) - len(really_new_candles)
        mfi_new_to = len(mfi)
        # if len(really_new_candles) == 0, then this for loop won't even start
        for i in range(mfi_new_from, mfi_new_to):
            mfi_i = mfi[i]
            logging.info(f"Current MFI value: {mfi_i}")
            logging.info(f"Current local minima: {last_local_minima}")
        
            # minima
            if mfi_i < MFI_THRESHOLD_LOW and mfi_i < last_local_minima:
                last_local_minima = mfi_i
                logging.info(f"New local minima: {last_local_minima}")
            
            diff_to_minima = mfi_i - last_local_minima
            
            if mfi_i > MFI_THRESHOLD_LOW_EXTENDED:
                # reset local minima only when mfi goes higher than extended threshold
                last_local_minima = 100
            
            if mfi_i < MFI_THRESHOLD_LOW_EXTENDED and diff_to_minima > MFI_STEP_THRESHOLD and not bought:
                last_local_minima = 100
                bought = True
                # buy signal
                order = execute_trade(symbol, quantity, "buy", config_path, dry_run)
                if order["price"] is None:
                    buy_price = float(candles[i][4]) # take last close price for dry run
                else:
                    buy_price = float(order["price"])
                buy_signals.append(i)
                logging.info(f"Buy signal: price {buy_price}, MFI {mfi_i}")
                break # this will break only from the mfi for loop - we can't sell inside this for loop because we would sell for the same price

            if not bought:
                logging.info(f"Not bought")
                continue

            # maxima
            current_mfi_threshold_high = MFI_THRESHOLD_HIGH - candles_since_buy * MFI_THRESHOLD_DECREASE_PER_CANDLE
            if current_mfi_threshold_high < MFI_THRESHOLD_HIGH_MIN:
                logging.info(f"MFI threshold reached minimum of {MFI_THRESHOLD_HIGH_MIN}")
                current_mfi_threshold_high = MFI_THRESHOLD_HIGH_MIN

            logging.info(f"Waiting for sell signal, candles_since_buy = {candles_since_buy}, current_mfi_threshold_high = {current_mfi_threshold_high}")

            if mfi_i > current_mfi_threshold_high:
                candles_above_threshold += 1
            else:
                candles_above_threshold = 0

            if candles_above_threshold >= 2:
                # sell as soon as MFI stays above threshold for 2 candles
                candles_above_threshold = 0
                candles_since_buy = 0
                bought = False
                # sell signal
                order = execute_trade(symbol, quantity, "sell", config_path, dry_run)

                if order["price"] is None:
                    sell_price = float(candles[i][4]) # take last close price for dry run
                else:
                    sell_price = float(order["price"])

                sell_signals.append(i)
                logging.info(f"Sell signal: price {sell_price}, MFI {mfi_i}")

                sell_final = sell_price * quantity
                buy_final = buy_price * quantity
                
                profit = sell_final - buy_final
                total_profit += profit
                profits.append(profit)
                logging.info(f"Current trade profit: {profit} USDT")
                logging.info(f"Total profit: {total_profit} USDT")
            else:
                candles_since_buy += 1 # bought, but not sold; we'll lower the MFI threshold every candle

        plot_asset({
            "symbol": symbol,
            "candles": candles,
            "mfi": mfi,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals
        }, f"_trading_{start_time_str}")

        # Sleep for 60 seconds before fetching new data
        logging.info(f"Waiting for the next candle, current candles above threshold: {candles_above_threshold}")
        time.sleep(60)
        
        # Get next candle and add it
        new_candles = get_new_candles_function(symbol, "1m", candles[-1][0])
        all_current_timestamps = [candle[0] for candle in candles]
        really_new_candles = [candle for candle in new_candles if candle[0] not in all_current_timestamps]
        if len(really_new_candles) == 0:
            # no new candles - can happen
            continue
        logging.info(f"Got {len(really_new_candles)} new candle(s)") 
        candles.extend(really_new_candles)
        
        # Check last N profits
        if len(profits) >= negative_cancel_num and all(p < 0 for p in profits[-negative_cancel_num:]):
            logging.info(f"Negative profit in last {negative_cancel_num} iterations. Exiting.")
            break

        # Check elapsed time - but finished only after trade was closed
        if not bought and (datetime.now() - start_time > timedelta(hours=MFI_TRADING_TIMEOUT_H)):
            logging.info(f"Running time exceeded {MFI_TRADING_TIMEOUT_H} hours. Exiting.")
            break

    logging.info(f"Finished. Total profit: {total_profit}")

    return({
            "symbol": symbol,
            "candles": candles,
            "mfi": mfi,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_profit": total_profit,
            "profits": profits
        })

