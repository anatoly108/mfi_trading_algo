import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta, timezone
from scipy.signal import find_peaks
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering without a display
import matplotlib.pyplot as plt
import mplfinance as mpf
import yaml
import time
import sys
from exchanges import Binance, Mexc
import multiprocessing
import signal
from multiprocessing import current_process, Manager

MFI_THRESHOLD_LOW = 20
MFI_THRESHOLD_LOW_EXTENDED = MFI_THRESHOLD_LOW + 10

MFI_THRESHOLD_HIGH = 80
MFI_THRESHOLD_DECREASE_PER_CANDLE = 2
MFI_THRESHOLD_HIGH_MIN = 50

MFI_STEP_THRESHOLD = 3
MFI_TIMEINTERVAL = 14
MFI_TRADING_TIMEOUT_H = 4
LOOKBACK_PERIOD_H = 24 # analysis is based on past 24h

VOL_THRESHOLD = 100e3

# Global termination flag
termination_flag = multiprocessing.Value('i', 0)

btc_liquidity_score_raw = None

# Signal handler to set the termination flag
def signal_handler(sig, frame):
    print(f"Signal {sig} received, setting termination flag")
    termination_flag.value = 1

signal.signal(signal.SIGTERM, signal_handler)

def setup_logging(log_dir=None, file_suffix="", log_to_stdout=True):
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'out')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{file_suffix}{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    # Clear existing logging handlers to prevent inheritance issues
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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

def get_exchange_client(exchange_name):
    if exchange_name == "binance":
        exchange_client = Binance("keys.yaml")
    elif exchange_name == "mexc":
        exchange_client = Mexc("keys.yaml")
    else:
        raise Exception(f"Incorrect exchange: {exchange_name}")

    logging.info(f"Exchange is set to: {exchange_client.__class__.__name__}")
    return exchange_client

def calculate_liquidity_score(symbol, exchange_client, is_setup=False):
    global btc_liquidity_score_raw
    if btc_liquidity_score_raw is None and not is_setup:
        # use BTC as the thing with highest liquidity
        btc_liquidity_score_raw = calculate_liquidity_score(symbol="BTCUSDT", exchange_client=exchange_client, is_setup=True)

    # Step 1: Get 24-hour ticker for trading volume
    ticker = exchange_client.get_ticker_data(symbol=symbol)
    trading_volume = float(ticker['quoteVolume'])  # This is the trading volume in USDT

    # Step 2: Calculate Market Capitalization
    # not available at the moment

    # Step 3: Calculate Order Book Depth (using 5% of the price range as an example)
    order_book = exchange_client.get_order_book(symbol=symbol, limit=200)
    bids = sum(float(bid[1]) for bid in order_book['bids'])  # Summing the volumes of buy orders
    asks = sum(float(ask[1]) for ask in order_book['asks'])  # Summing the volumes of sell orders
    order_book_depth = bids + asks  # Total order book depth

    # Step 4: Calculate Bid-Ask Spread
    best_bid = float(order_book['bids'][0][0])
    best_ask = float(order_book['asks'][0][0])
    bid_ask_spread = (best_ask - best_bid) / best_bid * 100  # Bid-ask spread in percentage

    # Step 5: Calculate Liquidity Score
    liquidity_score = (0.3 * trading_volume) + (0.5 * order_book_depth) + (0.2 * (1 / (1 + bid_ask_spread)))
    # original formula with market cap:
    # liquidity_score = (0.4 * trading_volume) + (0.3 * market_cap) + (0.1 * order_book_depth) + (0.2 * (1 / (1 + bid_ask_spread)))

    # Step 6: Normalization (optional, for a max score of 100)
    if is_setup:
        return liquidity_score
    
    normalized_score = (liquidity_score / btc_liquidity_score_raw) * 100
    return normalized_score

def usd_to_quantity(usdt_amount, current_price):
    initial_quantity = usdt_amount / current_price
    rounded_quantity = round(initial_quantity)
    
    # Calculate the actual USDT value after rounding
    resulting_usdt = rounded_quantity * current_price
    
    # Check if the difference is greater than 10%
    for digits_n in [1,2,3]: # allow max 3 digits after comma
        # allowed digits after comma depend on specific assets,
        # but this function will return any digits after comma only for expensive assets
        if abs(resulting_usdt - usdt_amount) > 0.1 * usdt_amount:
            rounded_quantity = round(initial_quantity, digits_n)
            resulting_usdt = rounded_quantity * current_price
        else:
            break
    
    return rounded_quantity

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
def get_candles(symbol, interval, exchange_client, startTime=None, endTime=None, hours=LOOKBACK_PERIOD_H, now=None):
    all_candles = []
    limit = 1000  # Maximum allowed by Binance

    if startTime is None and endTime is None and hours is not None:
        if now is None:
            now = get_last_complete_time_for_candles(interval)

        startTime = now - timedelta(hours=hours)
        endTime = now

    if startTime is None and endTime is None and hours is None:
        raise Exception("startTime is None and endTime is None and hours is None")

    startTimeUnix = convert_to_unix(startTime.replace(tzinfo=timezone.utc))
    endTimeUnix = convert_to_unix(endTime.replace(tzinfo=timezone.utc)) + 1

    num_candles = calculate_num_candles(interval, startTime, endTime)

    while len(all_candles) < num_candles:
        current_limit = min(limit, num_candles - len(all_candles))
        candles = exchange_client.get_candles(symbol=symbol,
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


def get_new_candles_from_exchange(symbol, interval, last_candle_timestamp, exchange_client):
    # Sleep for 60 seconds before fetching new data
    time.sleep(60)
    # get many candles just to be sure we didn't miss any due to some glitch
    return(get_candles(symbol=symbol, 
                        interval="1m", 
                        exchange_client=exchange_client,
                        startTime=datetime.fromtimestamp(last_candle_timestamp/1000, tz=timezone.utc) - timedelta(minutes=10), 
                        endTime=get_last_complete_time_for_candles(interval)))

def write_trading_results(results, global_results_csv, global_trades_csv, additional_values_to_add={}):
    keys_to_extract = ['symbol', 'total_profit', "total_profit_minus_fees"]
    result_dicts_for_data_frame = []
    for result in results:
        # Subset using dictionary comprehension
        result_sub = {key: result[key] for key in keys_to_extract if key in result}
        result_sub.update(additional_values_to_add)
        result_sub["date_start"] = result["candles"][0][0]
        result_sub["date_end"] = result["candles"][-1][0]
        result_dicts_for_data_frame.append(result_sub)

    results_df = pd.DataFrame.from_dict(result_dicts_for_data_frame)
    if os.path.exists(global_results_csv):
        header = False
    else:
        header = True
    results_df.to_csv(global_results_csv, mode="a", header=header, index=False)

    logging.info(f"Writing individual trades to files")
    trades = []
    for result in results:
        
        for i in range(len(result["buy_signals"])):
            buy_signal = result["buy_signals"][i]
            if len(result["sell_signals"]) < i + 1:
                # buy without sell - can happen in analysis
                break 

            sell_signal = result["sell_signals"][i]
            
            mfi_i_buy = result["mfi"][buy_signal]
            mfi_i_sell = result["mfi"][sell_signal]

            profit = result["profits"][i]

            candle_buy = result["candles"][buy_signal]
            candle_buy_time = candle_buy[0]
            candle_buy_close = candle_buy[4]
            candle_sell = result["candles"][sell_signal]
            candle_sell_time = candle_sell[0]
            candle_sell_close = candle_sell[4]

            trade_dict = {
                "symbol": result["symbol"],
                "i": i,
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "mfi_i_buy": mfi_i_buy,
                "mfi_i_sell": mfi_i_sell,
                "profit": profit,
                "candle_buy_time": candle_buy_time,
                "candle_buy_close": candle_buy_close,
                "candle_sell_time": candle_sell_time,
                "candle_sell_close": candle_sell_close
            }
            trade_dict.update(additional_values_to_add)
            trades.append(trade_dict)
    
    trades_df = pd.DataFrame.from_dict(trades)
    if os.path.exists(global_trades_csv):
        header = False
    else:
        header = True
    trades_df.to_csv(global_trades_csv, mode="a", header=header, index=False)

def run_mfi_trading_algo(symbol, dry_run, exchange_client,
                         negative_cancel_num=3, get_new_candles_function=get_new_candles_from_exchange,
                         candles = None, exit_after_no_candle=False, do_plot=True, out_dir="out", 
                         quantity=None, usdt_amount=None): 
    if quantity is None and usdt_amount is None:
        raise Exception("quantity is None and usdt_amount is None")

    ticker = exchange_client.get_ticker_data(symbol=symbol)
    current_price = float(ticker['lastPrice'])
    
    if usdt_amount is not None:
        quantity = usd_to_quantity(usdt_amount, current_price)
    
    usdt_amount_final = current_price * quantity
    logging.info(f"Chosen quantity is: {quantity}, equivalent to {usdt_amount_final} USDT")

    start_time = datetime.now()
    start_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    # Load initial candles and MFI
    if candles is None:
        candles = get_candles(symbol=symbol, interval="1m", exchange_client=exchange_client)

    mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)

    last_local_minima = 100
    candles_above_threshold = 0
    bought = False
    really_new_candles = []
    total_profit = 0
    total_profit_minus_fees = 0
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
        if quantity == 0:
            # can happen if asset is expensive and usdt_amount is small
            logging.warning(f"Quantity equals 0. Breaking.")
            # this if can also be outside of this loop (before it),
            # but it's more convenient to have it here because then we won't have to
            # repeat the part of the function after the loop 
            break
            
        if termination_flag.value:
            logging.warning(f"Process {current_process().pid}: Termination requested, finishing up.")
            if bought:
                logging.warning("Termination requested, but not sold yet. Waiting for sell signal.")
            else:
                logging.warning("Termination requested, not bought. Breaking.")
                break

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
                order = exchange_client.execute_market_order(symbol=symbol, side="buy", quantity=quantity, dry_run=dry_run)
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
                order = exchange_client.execute_market_order(symbol=symbol, side="sell", quantity=quantity, dry_run=dry_run)

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
            }, f"_trading_{start_time_str}_{exchange_client.__class__.__name__}", out_dir=out_dir)

        logging.info(f"Waiting for the next candle, current candles above threshold: {candles_above_threshold}")
        
        # Get next candle and add it
        last_candle_timestamp = candles[-1][0]
        last_candle_datetime_obj = datetime.fromtimestamp(last_candle_timestamp/1000, tz=timezone.utc)
        logging.info(f"Last candle time before new candles, UNIX: {last_candle_timestamp}, UTC: {last_candle_datetime_obj}")
        
        new_candles = get_new_candles_function(symbol=symbol, interval="1m", last_candle_timestamp=last_candle_timestamp, exchange_client=exchange_client)
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

    total_profit_minus_fees = total_profit - (len(buy_signals) + len(sell_signals))*usdt_amount_final*exchange_client.get_taker_fee_fraction()

    logging.info(f"Finished. Total profit: {total_profit}, minus fees: {total_profit_minus_fees}")

    res_dict = {
            "symbol": symbol,
            "candles": candles,
            "mfi": mfi,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_profit": total_profit,
            "total_profit_minus_fees": total_profit_minus_fees,
            "profits": profits
        }

    return(res_dict)

