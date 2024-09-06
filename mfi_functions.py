import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta, timezone
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import yaml
import time
import sys
from exchanges import Binance, Mexc

MFI_THRESHOLD_LOW = 20
MFI_THRESHOLD_LOW_EXTENDED = MFI_THRESHOLD_LOW + 10

MFI_THRESHOLD_HIGH = 80
MFI_THRESHOLD_DECREASE_PER_CANDLE = 2
MFI_THRESHOLD_HIGH_MIN = 50

MFI_STEP_THRESHOLD = 3
MFI_TIMEINTERVAL = 14
MFI_TRADING_TIMEOUT_H = 2 # default is 12, 2 is for testing

VOL_THRESHOLD = 100e3

ExchangeClient = Binance("keys.yaml")

def setup_logging(log_dir=None, file_suffix="", log_to_stdout=True):
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'out')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{file_suffix}{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    handlers = [logging.FileHandler(log_filepath)]  # Log to file by default

    if log_to_stdout:
        handlers.append(logging.StreamHandler())  # Log to console if requested

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Define a custom exception hook to log uncaught exceptions
    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log KeyboardInterrupt to avoid log spam when user interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Set the custom exception hook as the global one
    sys.excepthook = log_exception



def usd_to_quantity(usdt_amount, current_price):
    return(round(usdt_amount / current_price))

def convert_to_unix(date_obj):
    return int(date_obj.timestamp() * 1000)

def get_last_complete_time_for_candles(interval):
    now = datetime.now(timezone.utc)
    
    if interval.endswith('m'):  # Minute intervals
        minutes = int(interval[:-1])
        last_complete_time = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % minutes)
    
    elif interval.endswith('h'):  # Hour intervals
        print("warning: untested interval")
        hours = int(interval[:-1])
        last_complete_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % hours)
    
    elif interval == '1d':  # 1 day interval
        print("warning: untested interval")
        last_complete_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    
    elif interval == '3d':  # 3 days interval
        print("warning: untested interval")
        last_complete_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(now.day - 1) % 3)
    
    elif interval == '1w':  # 1 week interval
        print("warning: untested interval")
        days_since_monday = now.weekday()
        last_complete_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
    
    elif interval == '1M':  # 1 month interval
        raise ValueError("Unsupported interval")
    else:
        raise ValueError("Unsupported interval")
    
    return last_complete_time

def calculate_num_candles(interval, startTime, endTime):
    interval_seconds = 0
    if interval.endswith('m'):  # Minute intervals
        interval_seconds = int(interval[:-1]) * 60
    elif interval.endswith('h'):  # Hour intervals
        interval_seconds = int(interval[:-1]) * 3600
    elif interval == '1d':  # 1 day interval
        interval_seconds = 86400
    elif interval == '3d':  # 3 days interval
        interval_seconds = 86400 * 3
    elif interval == '1w':  # 1 week interval
        interval_seconds = 86400 * 7
    elif interval == '1M':  # 1 month interval
        # This is more complex since months vary in length, but you can use an average or approximate value.
        # Here's an average month length in seconds:
        interval_seconds = 86400 * 30.44
    else:
        raise ValueError("Unsupported interval")

    total_seconds = (endTime - startTime).total_seconds()
    num_candles = int(total_seconds // interval_seconds)
    return num_candles


# note: this will return only complete candles!
# startTime and endTime are datetime objects, easiest way to specify: datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
def get_candles(symbol, interval, startTime=None, endTime=None):
    all_candles = []
    limit = 1000  # Maximum allowed by Binance

    if startTime is None and endTime is None:
        now = get_last_complete_time_for_candles(interval)
        startTime = now - timedelta(hours=24)
        endTime = now

    startTimeUnix = convert_to_unix(startTime.replace(tzinfo=timezone.utc))
    endTimeUnix = convert_to_unix(endTime.replace(tzinfo=timezone.utc)) + 1

    num_candles = calculate_num_candles(interval, startTime, endTime)

    while len(all_candles) < num_candles:
        current_limit = min(limit, num_candles - len(all_candles))
        candles = ExchangeClient.get_candles(symbol=symbol,
                                             interval=interval,
                                             startTime=startTimeUnix,
                                             endTime=endTimeUnix,
                                             limit=current_limit) # Fetch only the remaining needed candles
        
        if not candles:
            break
        
        all_candles.extend(candles)
        
        # Move startTime forward to the last candle's timestamp + 1ms to avoid overlapping
        last_candle_time = candles[-1][0]
        startTimeUnix = last_candle_time + 1
        
        # To break if all possible candles were returned
        if len(candles) < limit:
            break
    
    # Sort all candles by timestamp to ensure correct order
    all_candles = sorted(all_candles, key=lambda x: x[0])
    
    return all_candles


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


def plot_asset(asset_data, plot_suffix="", out_dir="out"):
    symbol = asset_data["symbol"]
    candles = asset_data["candles"]
    mfi = asset_data["mfi"]
    buy_signals = asset_data["buy_signals"]
    sell_signals = asset_data["sell_signals"]

    # Convert candles to a DataFrame
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/{symbol}_chart{plot_suffix}.png')
    # plt.show()
    plt.close()


def get_new_candles_from_exchange(symbol, interval, last_candle_timestamp):
    # Sleep for 60 seconds before fetching new data
    time.sleep(60)
    # get many candles just to be sure we didn't miss any due to some glitch
    return(get_candles(symbol=symbol, 
                        interval="1m", 
                        startTime=datetime.fromtimestamp(last_candle_timestamp/1000, tz=timezone.utc) - timedelta(minutes=10), 
                        endTime=get_last_complete_time_for_candles(interval)))

def run_mfi_trading_algo(symbol, dry_run, 
                         negative_cancel_num=3, get_new_candles_function=get_new_candles_from_exchange,
                         candles = None, exit_after_no_candle=False, do_plot=True, out_dir="out", quantity=None, usdt_amount=None): 

    if quantity is None and usdt_amount is None:
        raise Exception("quantity is None and usdt_amount is None")

    ticker = ExchangeClient.get_ticker_data(symbol=symbol)
    current_price = float(ticker['lastPrice'])
    
    if usdt_amount is not None:
        quantity = usd_to_quantity(usdt_amount, current_price)
    
    usdt_amount_final = current_price * quantity
    logging.info(f"Chosen quantity is: {quantity}, equivalent to {usdt_amount_final} USDT")

    start_time = datetime.now()
    start_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    # Load initial candles and MFI
    if candles is None:
        candles = get_candles(symbol, "1m")

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
    iteration = 0

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
        iteration += 1

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
                order = ExchangeClient.execute_market_order(symbol=symbol, side="buy", quantity=quantity, dry_run=dry_run)
                if order["price"] is None:
                    buy_price = candles[i][4] # take last close price for dry run
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
                order = ExchangeClient.execute_market_order(symbol=symbol, side="sell", quantity=quantity, dry_run=dry_run)

                if order["price"] is None:
                    sell_price = candles[i][4] # take last close price for dry run
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

        if do_plot:
            plot_asset({
                "symbol": symbol,
                "candles": candles,
                "mfi": mfi,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            }, f"_trading_{start_time_str}", out_dir=out_dir)

        logging.info(f"Waiting for the next candle, current candles above threshold: {candles_above_threshold}")
        
        # Get next candle and add it
        last_candle_timestamp = candles[-1][0]
        last_candle_datetime_obj = datetime.fromtimestamp(last_candle_timestamp/1000, tz=timezone.utc)
        logging.info(f"Last candle time before new candles, UNIX: {last_candle_timestamp}, UTC: {last_candle_datetime_obj}")
        
        new_candles = get_new_candles_function(symbol, "1m", last_candle_timestamp)
        all_current_timestamps = [candle[0] for candle in candles]
        really_new_candles = [candle for candle in new_candles if candle[0] not in all_current_timestamps]

        logging.info(f"Got {len(really_new_candles)} new candle(s)")
        if len(really_new_candles) == 0 and exit_after_no_candle:
            logging.info(f"No new candles and exit_after_no_candle is True. Exiting.")
            break

        if len(really_new_candles) == 0:
            # no new candles - can happen
            continue
        
        candles.extend(really_new_candles)
        last_candle_timestamp = candles[-1][0]
        last_candle_datetime_obj = datetime.fromtimestamp(last_candle_timestamp/1000, tz=timezone.utc)
        logging.info(f"Last candle time after new candles, UNIX: {last_candle_timestamp}, UTC: {last_candle_datetime_obj}")

        # Check last N profits
        if len(profits) >= negative_cancel_num and all(p < 0 for p in profits[-negative_cancel_num:]):
            logging.info(f"Negative profit in last {negative_cancel_num} iterations. Exiting.")
            break

        # Check elapsed time - but finished only after trade was closed
        if not bought and (datetime.now() - start_time > timedelta(hours=MFI_TRADING_TIMEOUT_H)):
            logging.info(f"Running time exceeded {MFI_TRADING_TIMEOUT_H} hours. Exiting.")
            break

    total_profit_minus_fees = (len(buy_signals) + len(sell_signals))*usdt_amount_final*ExchangeClient.get_taker_fee_fraction()

    logging.info(f"Finished. Total profit: {total_profit}, minus fees: {total_profit_minus_fees}")

    return({
            "symbol": symbol,
            "candles": candles,
            "mfi": mfi,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_profit": total_profit,
            "total_profit_minus_fees": total_profit_minus_fees,
            "profits": profits
        })

